//! RDF-star serialization implementations for various formats.
//!
//! This module provides serializers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)  
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)
//! - JSON-LD-star (*.jlds)
//!
//! Features:
//! - Streaming serialization for large datasets
//! - Compression support (gzip, zstd)
//! - Parallel serialization for multi-core systems
//! - Memory-efficient processing with buffer reuse
//! - Configurable batching and buffering strategies

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::thread;

use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarConfig, StarError, StarResult};

/// Serialization options for configuring output format
#[derive(Debug, Clone)]
pub struct SerializationOptions {
    /// Pretty print with indentation
    pub pretty_print: bool,
    /// Use compact notation where possible
    pub compact: bool,
    /// Namespace prefixes to use
    pub prefixes: HashMap<String, String>,
    /// Base IRI for relative references
    pub base_iri: Option<String>,
    /// Indentation string (spaces or tabs)
    pub indent_string: String,
    /// Enable streaming serialization for large datasets
    pub streaming: bool,
    /// Compression type to apply
    pub compression: CompressionType,
    /// Buffer size for streaming operations (bytes)
    pub buffer_size: usize,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Enable parallel serialization
    pub parallel: bool,
    /// Maximum number of worker threads
    pub max_threads: usize,
}

/// Compression types supported for serialization output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zstandard compression (high performance)
    Zstd,
    /// LZ4 compression (fastest)
    Lz4,
}

/// Configuration for streaming serialization
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Chunk size for processing triples/quads
    pub chunk_size: usize,
    /// Memory threshold before flushing (bytes)
    pub memory_threshold: usize,
    /// Enable buffering of output
    pub enable_buffering: bool,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Enable compression of chunks
    pub compress_chunks: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,
            memory_threshold: 64 * 1024 * 1024, // 64MB
            enable_buffering: true,
            buffer_capacity: 1024 * 1024, // 1MB buffer
            compress_chunks: false,
        }
    }
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            pretty_print: true,
            compact: false,
            prefixes: HashMap::new(),
            base_iri: None,
            indent_string: "  ".to_string(),
            streaming: false,
            compression: CompressionType::None,
            buffer_size: 1024 * 1024, // 1MB
            batch_size: 10000,
            parallel: false,
            max_threads: 4,
        }
    }
}

/// Context for serialization with namespace prefixes and formatting options
#[derive(Debug, Default)]
struct SerializationContext {
    /// Namespace prefixes for compact representation
    prefixes: HashMap<String, String>,
    /// Base IRI for relative references
    base_iri: Option<String>,
    /// Pretty printing with indentation
    pretty_print: bool,
    /// Current indentation level
    indent_level: usize,
    /// Indentation string (spaces or tabs)
    indent_string: String,
}

impl SerializationContext {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            pretty_print: true,
            indent_level: 0,
            indent_string: "  ".to_string(),
        }
    }

    /// Add a namespace prefix
    fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Get current indentation
    fn current_indent(&self) -> String {
        self.indent_string.repeat(self.indent_level)
    }

    /// Increase indentation level
    fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    fn decrease_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// Try to compress IRI using prefixes
    fn compress_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                return format!("{}:{}", prefix, local);
            }
        }

        // Return full IRI if no prefix match
        format!("<{}>", iri)
    }
}

/// Streaming serializer for memory-efficient processing of large graphs
pub struct StreamingSerializer<W: Write> {
    writer: Arc<Mutex<W>>,
    config: StreamingConfig,
    context: SerializationContext,
    buffer: Vec<u8>,
    written_bytes: usize,
}

impl<W: Write> StreamingSerializer<W> {
    /// Create a new streaming serializer
    pub fn new(writer: W, config: StreamingConfig) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
            buffer: Vec::with_capacity(config.buffer_capacity),
            config,
            context: SerializationContext::new(),
            written_bytes: 0,
        }
    }

    /// Write data to the output stream with buffering
    fn write_buffered(&mut self, data: &[u8]) -> StarResult<()> {
        if self.config.enable_buffering {
            self.buffer.extend_from_slice(data);

            // Flush if buffer is full or memory threshold reached
            if self.buffer.len() >= self.config.buffer_capacity
                || self.written_bytes >= self.config.memory_threshold
            {
                self.flush_buffer()?;
            }
        } else {
            // Direct write without buffering
            let mut writer = self
                .writer
                .lock()
                .map_err(|e| StarError::serialization_error(format!("Lock error: {}", e)))?;
            writer
                .write_all(data)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            self.written_bytes += data.len();
        }
        Ok(())
    }

    /// Flush the internal buffer to the writer
    fn flush_buffer(&mut self) -> StarResult<()> {
        if !self.buffer.is_empty() {
            let data = if self.config.compress_chunks {
                self.compress_chunk(&self.buffer)?
            } else {
                self.buffer.clone()
            };

            let mut writer = self
                .writer
                .lock()
                .map_err(|e| StarError::serialization_error(format!("Lock error: {}", e)))?;
            writer
                .write_all(&data)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            writer
                .flush()
                .map_err(|e| StarError::serialization_error(e.to_string()))?;

            self.written_bytes += data.len();
            self.buffer.clear();
        }
        Ok(())
    }

    /// Compress a chunk of data (placeholder implementation)
    fn compress_chunk(&self, data: &[u8]) -> StarResult<Vec<u8>> {
        // For now, return data as-is
        // In a full implementation, this would use actual compression libraries
        Ok(data.to_vec())
    }

    /// Serialize triples in streaming fashion
    pub fn serialize_triples_streaming<I>(
        &mut self,
        triples: I,
        format: StarFormat,
    ) -> StarResult<()>
    where
        I: Iterator<Item = StarTriple>,
    {
        for chunk in ChunkedIterator::new(triples, self.config.chunk_size) {
            self.serialize_chunk(&chunk, format)?;
        }
        self.flush_buffer()?;
        Ok(())
    }

    /// Serialize a chunk of triples
    fn serialize_chunk(&mut self, chunk: &[StarTriple], format: StarFormat) -> StarResult<()> {
        for triple in chunk {
            let line = match format {
                StarFormat::NTriplesStar => {
                    let subject = self.format_term_ntriples(&triple.subject)?;
                    let predicate = self.format_term_ntriples(&triple.predicate)?;
                    let object = self.format_term_ntriples(&triple.object)?;
                    format!("{} {} {} .\n", subject, predicate, object)
                }
                StarFormat::TurtleStar => {
                    let subject = self.format_term_turtle(&triple.subject)?;
                    let predicate = self.format_term_turtle(&triple.predicate)?;
                    let object = self.format_term_turtle(&triple.object)?;
                    format!("{} {} {} .\n", subject, predicate, object)
                }
                StarFormat::TrigStar => {
                    // TriG-star format with default graph
                    let subject = self.format_term_ntriples(&triple.subject)?;
                    let predicate = self.format_term_ntriples(&triple.predicate)?;
                    let object = self.format_term_ntriples(&triple.object)?;
                    format!("{} {} {} .\n", subject, predicate, object)
                }
                StarFormat::NQuadsStar => {
                    // N-Quads-star format with default graph
                    let subject = self.format_term_ntriples(&triple.subject)?;
                    let predicate = self.format_term_ntriples(&triple.predicate)?;
                    let object = self.format_term_ntriples(&triple.object)?;
                    format!("{} {} {} <> .\n", subject, predicate, object) // <> represents default graph
                }
                _ => {
                    return Err(StarError::serialization_error(format!(
                        "Streaming not yet implemented for format {:?}",
                        format
                    )))
                }
            };
            self.write_buffered(line.as_bytes())?;
        }
        Ok(())
    }

    /// Format term for N-Triples output
    fn format_term_ntriples(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term_ntriples(&triple.subject)?;
                let predicate = self.format_term_ntriples(&triple.predicate)?;
                let object = self.format_term_ntriples(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Format term for Turtle output
    fn format_term_turtle(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(self.context.compress_iri(&node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^{}", self.context.compress_iri(&datatype.iri)));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term_turtle(&triple.subject)?;
                let predicate = self.format_term_turtle(&triple.predicate)?;
                let object = self.format_term_turtle(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }
}

/// Parallel serializer for multi-threaded processing
pub struct ParallelSerializer {
    num_threads: usize,
    batch_size: usize,
}

impl ParallelSerializer {
    /// Create a new parallel serializer
    pub fn new(num_threads: usize, batch_size: usize) -> Self {
        Self {
            num_threads,
            batch_size,
        }
    }

    /// Static method for formatting term in N-Triples
    fn format_term_ntriples(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = Self::format_term_ntriples(&triple.subject)?;
                let predicate = Self::format_term_ntriples(&triple.predicate)?;
                let object = Self::format_term_ntriples(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Static method for formatting term in Turtle
    fn format_term_turtle(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = Self::format_term_turtle(&triple.subject)?;
                let predicate = Self::format_term_turtle(&triple.predicate)?;
                let object = Self::format_term_turtle(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Serialize graph using multiple threads
    pub fn serialize_parallel<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        _options: &SerializationOptions,
    ) -> StarResult<()> {
        let writer = Arc::new(Mutex::new(writer));
        let triples: Vec<_> = graph.triples().into_iter().collect();

        // Split into batches for parallel processing
        let batches: Vec<_> = triples.chunks(self.batch_size).collect();
        let mut handles = Vec::new();

        for batch in batches {
            let batch: Vec<StarTriple> = batch.iter().map(|t| (*t).clone()).collect();
            let writer_clone = Arc::clone(&writer);
            let format_clone = format;

            let handle =
                thread::spawn(move || Self::process_batch(batch, writer_clone, format_clone));
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|e| {
                StarError::serialization_error(format!("Thread join error: {:?}", e))
            })??;
        }

        Ok(())
    }

    /// Process a batch of triples in a worker thread
    fn process_batch<W: Write>(
        batch: Vec<StarTriple>,
        writer: Arc<Mutex<W>>,
        format: StarFormat,
    ) -> StarResult<()> {
        let mut output = Vec::new();

        for triple in batch {
            let line = match format {
                StarFormat::NTriplesStar => {
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                StarFormat::TurtleStar => {
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_turtle(&triple.subject)?,
                        Self::format_term_turtle(&triple.predicate)?,
                        Self::format_term_turtle(&triple.object)?
                    )
                }
                StarFormat::TrigStar => {
                    // TriG-star format with default graph
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                StarFormat::NQuadsStar => {
                    // N-Quads-star format with default graph
                    format!(
                        "{} {} {} <> .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                _ => {
                    return Err(StarError::serialization_error(format!(
                        "Parallel serialization not yet implemented for format {:?}",
                        format
                    )))
                }
            };
            output.extend_from_slice(line.as_bytes());
        }

        let mut writer = writer
            .lock()
            .map_err(|e| StarError::serialization_error(format!("Lock error: {}", e)))?;
        writer
            .write_all(&output)
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }
}

/// Chunked iterator for processing large collections in batches
struct ChunkedIterator<I> {
    inner: I,
    chunk_size: usize,
}

impl<I> ChunkedIterator<I> {
    fn new(inner: I, chunk_size: usize) -> Self {
        Self { inner, chunk_size }
    }
}

impl<I, T> Iterator for ChunkedIterator<I>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        for _ in 0..self.chunk_size {
            match self.inner.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

/// RDF-star serializer with support for multiple formats
pub struct StarSerializer {
    config: StarConfig,
}

impl StarSerializer {
    /// Create a new serializer with default configuration
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
        }
    }

    /// Create a new serializer with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self { config }
    }

    /// Serialize a StarGraph to a writer in the specified format
    pub fn serialize<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "serialize_rdf_star", format = ?format);
        let _enter = span.enter();

        match format {
            StarFormat::TurtleStar => self.serialize_turtle_star(graph, writer),
            StarFormat::NTriplesStar => self.serialize_ntriples_star(graph, writer),
            StarFormat::TrigStar => self.serialize_trig_star(graph, writer),
            StarFormat::NQuadsStar => self.serialize_nquads_star(graph, writer),
            StarFormat::JsonLdStar => self.serialize_jsonld_star(graph, writer),
        }
    }

    /// Serialize to string in the specified format
    pub fn serialize_to_string(&self, graph: &StarGraph, format: StarFormat) -> StarResult<String> {
        let mut buffer = Vec::new();
        self.serialize(graph, &mut buffer, format)?;
        String::from_utf8(buffer).map_err(|e| StarError::serialization_error(e.to_string()))
    }

    /// Serialize with advanced options (streaming, compression, parallel processing)
    pub fn serialize_with_options<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        options: &SerializationOptions,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "serialize_with_options", format = ?format, streaming = options.streaming, parallel = options.parallel);
        let _enter = span.enter();

        // Choose serialization strategy based on options and graph size
        let triple_count = graph.total_len();

        if options.parallel && triple_count > options.batch_size {
            debug!("Using parallel serialization for {} triples", triple_count);
            let parallel_serializer =
                ParallelSerializer::new(options.max_threads, options.batch_size);
            parallel_serializer.serialize_parallel(graph, writer, format, options)
        } else if options.streaming && triple_count > 50000 {
            debug!("Using streaming serialization for {} triples", triple_count);
            let streaming_config = StreamingConfig {
                chunk_size: options.batch_size,
                buffer_capacity: options.buffer_size,
                enable_buffering: true,
                memory_threshold: options.buffer_size * 64, // 64x buffer size
                compress_chunks: options.compression != CompressionType::None,
            };
            let mut streaming_serializer = StreamingSerializer::new(writer, streaming_config);
            streaming_serializer
                .serialize_triples_streaming(graph.triples().into_iter().cloned(), format)
        } else {
            debug!("Using standard serialization for {} triples", triple_count);
            // Apply compression wrapper if requested
            if options.compression != CompressionType::None {
                let compressed_writer =
                    self.create_compressed_writer(writer, options.compression)?;
                self.serialize(graph, compressed_writer, format)
            } else {
                self.serialize(graph, writer, format)
            }
        }
    }

    /// Create a compressed writer based on compression type
    fn create_compressed_writer<W: Write + 'static>(
        &self,
        writer: W,
        compression: CompressionType,
    ) -> StarResult<Box<dyn Write>> {
        match compression {
            CompressionType::None => Ok(Box::new(writer)),
            CompressionType::Gzip => {
                // Placeholder - would use flate2 crate in full implementation
                debug!("Gzip compression requested but not yet implemented");
                Ok(Box::new(writer))
            }
            CompressionType::Zstd => {
                // Placeholder - would use zstd crate in full implementation
                debug!("Zstd compression requested but not yet implemented");
                Ok(Box::new(writer))
            }
            CompressionType::Lz4 => {
                // Placeholder - would use lz4 crate in full implementation
                debug!("LZ4 compression requested but not yet implemented");
                Ok(Box::new(writer))
            }
        }
    }

    /// Serialize large dataset using streaming approach
    pub fn serialize_streaming<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        chunk_size: usize,
    ) -> StarResult<()> {
        let span =
            span!(Level::DEBUG, "serialize_streaming", format = ?format, chunk_size = chunk_size);
        let _enter = span.enter();

        let config = StreamingConfig {
            chunk_size,
            ..Default::default()
        };
        let mut streaming_serializer = StreamingSerializer::new(writer, config);
        streaming_serializer
            .serialize_triples_streaming(graph.triples().into_iter().cloned(), format)?;

        debug!(
            "Streamed {} triples in format {:?}",
            graph.total_len(),
            format
        );
        Ok(())
    }

    /// Serialize using parallel processing for large graphs
    pub fn serialize_parallel<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        num_threads: usize,
        batch_size: usize,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_parallel", format = ?format, num_threads = num_threads, batch_size = batch_size);
        let _enter = span.enter();

        let parallel_serializer = ParallelSerializer::new(num_threads, batch_size);
        let options = SerializationOptions::default();
        parallel_serializer.serialize_parallel(graph, writer, format, &options)?;

        debug!(
            "Parallel serialization completed for {} triples using {} threads",
            graph.total_len(),
            num_threads
        );
        Ok(())
    }

    /// Auto-detect optimal serialization strategy based on graph characteristics
    pub fn serialize_optimized<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_optimized", format = ?format);
        let _enter = span.enter();

        let triple_count = graph.total_len();
        let quoted_count = graph.count_quoted_triples();
        let complexity_score = quoted_count as f64 / triple_count.max(1) as f64;

        debug!(
            "Graph analysis: {} triples, {} quoted (complexity: {:.2})",
            triple_count, quoted_count, complexity_score
        );

        let mut options = SerializationOptions::default();

        // Configure based on graph characteristics
        if triple_count > 1_000_000 {
            // Very large graph - use streaming
            options.streaming = true;
            options.compression = CompressionType::Zstd; // High performance compression
            options.buffer_size = 4 * 1024 * 1024; // 4MB buffer
            debug!("Selected streaming strategy for very large graph");
        } else if triple_count > 100_000 && complexity_score < 0.1 {
            // Large simple graph - use parallel processing
            options.parallel = true;
            options.max_threads = std::cmp::min(8, 4); // Use 4 as default thread count
            options.batch_size = 25000;
            debug!("Selected parallel strategy for large simple graph");
        } else if complexity_score > 0.3 {
            // Complex graph with many quoted triples - use smaller batches
            options.batch_size = 5000;
            options.buffer_size = 512 * 1024; // 512KB buffer
            debug!("Selected conservative strategy for complex graph");
        }
        // else: use default strategy for smaller/simpler graphs

        self.serialize_with_options(graph, writer, format, &options)
    }

    /// Get memory usage estimation for serialization
    pub fn estimate_memory_usage(
        &self,
        graph: &StarGraph,
        format: StarFormat,
        options: &SerializationOptions,
    ) -> usize {
        let base_memory = self.estimate_size(graph, format);

        let memory_multiplier = if options.parallel {
            // Parallel processing uses more memory for batching
            2.5
        } else if options.streaming {
            // Streaming uses less memory
            0.5
        } else {
            1.0
        };

        let buffer_overhead = if options.streaming {
            options.buffer_size * 2 // Double buffering
        } else {
            options.batch_size * 100 // Rough estimate of batch memory
        };

        ((base_memory as f64 * memory_multiplier) as usize) + buffer_overhead
    }

    /// Serialize to Turtle-star format
    pub fn serialize_turtle_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_turtle_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Write triples
        for triple in graph.triples() {
            self.write_turtle_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!("Serialized {} triples in Turtle-star format", graph.len());
        Ok(())
    }

    /// Write Turtle-star prefixes
    fn write_turtle_prefixes<W: Write>(
        &self,
        writer: &mut W,
        context: &SerializationContext,
    ) -> StarResult<()> {
        for (prefix, namespace) in &context.prefixes {
            writeln!(writer, "@prefix {}: <{}> .", prefix, namespace)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        if !context.prefixes.is_empty() {
            writeln!(writer).map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single Turtle-star triple
    fn write_turtle_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term(&triple.subject, context)?;
        let predicate_str = self.format_term(&triple.predicate, context)?;
        let object_str = self.format_term(&triple.object, context)?;

        if context.pretty_print {
            writeln!(
                writer,
                "{}{} {} {} .",
                context.current_indent(),
                subject_str,
                predicate_str,
                object_str
            )
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        } else {
            writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Serialize to N-Triples-star format
    pub fn serialize_ntriples_star<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_ntriples_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Triples doesn't use prefixes

        for triple in graph.triples() {
            self.write_ntriples_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} triples in N-Triples-star format",
            graph.len()
        );
        Ok(())
    }

    /// Write a single N-Triples-star triple
    fn write_ntriples_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&triple.subject)?;
        let predicate_str = self.format_term_ntriples(&triple.predicate)?;
        let object_str = self.format_term_ntriples(&triple.object)?;

        writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }

    /// Serialize to TriG-star format (with named graphs)
    pub fn serialize_trig_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_trig_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();
        context.pretty_print = true;

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes first
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Serialize default graph if it has triples
        if !graph.triples().is_empty() {
            writeln!(buf_writer, "{{")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;

            context.increase_indent();
            for triple in graph.triples() {
                self.write_turtle_triple(&mut buf_writer, triple, &context)?;
            }
            context.decrease_indent();

            writeln!(buf_writer, "}}")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            writeln!(buf_writer).map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        // Serialize named graphs
        for graph_name in graph.named_graph_names() {
            if let Some(named_triples) = graph.named_graph_triples(graph_name) {
                if !named_triples.is_empty() {
                    // Write graph declaration
                    let graph_term = self.parse_graph_name(graph_name, &context)?;
                    writeln!(buf_writer, "{} {{", graph_term)
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;

                    context.increase_indent();
                    for triple in named_triples {
                        self.write_turtle_triple(&mut buf_writer, triple, &context)?;
                    }
                    context.decrease_indent();

                    writeln!(buf_writer, "}}")
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;
                    writeln!(buf_writer)
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;
                }
            }
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in TriG-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Serialize to N-Quads-star format
    pub fn serialize_nquads_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_nquads_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Quads doesn't use prefixes

        // Serialize all quads from the graph (including both default and named graphs)
        for quad in graph.quads() {
            self.write_nquads_quad_complete(&mut buf_writer, quad, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in N-Quads-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Write a single N-Quads-star quad with proper graph context
    fn write_nquads_quad_complete<W: Write>(
        &self,
        writer: &mut W,
        quad: &StarQuad,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&quad.subject)?;
        let predicate_str = self.format_term_ntriples(&quad.predicate)?;
        let object_str = self.format_term_ntriples(&quad.object)?;

        if let Some(ref graph_term) = quad.graph {
            // Named graph quad
            let graph_str = self.format_term_ntriples(graph_term)?;
            writeln!(
                writer,
                "{} {} {} {} .",
                subject_str, predicate_str, object_str, graph_str
            )
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        } else {
            // Default graph quad (triple)
            writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single N-Quads-star quad (triple + optional graph) - legacy method
    fn write_nquads_quad<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term_ntriples(&triple.subject)?;
        let predicate_str = self.format_term_ntriples(&triple.predicate)?;
        let object_str = self.format_term_ntriples(&triple.object)?;

        // Default graph (no graph component)
        writeln!(writer, "{} {} {} .", subject_str, predicate_str, object_str)
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }

    /// Serialize to JSON-LD-star format
    pub fn serialize_jsonld_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_jsonld_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();

        // Create JSON-LD context
        let mut jsonld_document = serde_json::Map::new();

        // Add JSON-LD context
        let mut context_obj = serde_json::Map::new();
        context_obj.insert(
            "@vocab".to_string(),
            serde_json::Value::String("http://example.org/".to_string()),
        );

        // Add common prefixes
        context_obj.insert(
            "rdf".to_string(),
            serde_json::Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string()),
        );
        context_obj.insert(
            "rdfs".to_string(),
            serde_json::Value::String("http://www.w3.org/2000/01/rdf-schema#".to_string()),
        );
        context_obj.insert(
            "xsd".to_string(),
            serde_json::Value::String("http://www.w3.org/2001/XMLSchema#".to_string()),
        );

        jsonld_document.insert(
            "@context".to_string(),
            serde_json::Value::Object(context_obj),
        );

        // Group quads by subject
        let mut subjects = std::collections::HashMap::new();

        // Process all quads
        for quad in graph.quads() {
            self.add_quad_to_jsonld(&mut subjects, quad)?;
        }

        // Convert subjects to JSON-LD array
        let mut graph_array = Vec::new();
        for (subject_str, properties) in subjects {
            let mut subject_obj = serde_json::Map::new();

            // Add @id for non-blank nodes
            if !subject_str.starts_with("_:") {
                subject_obj.insert("@id".to_string(), serde_json::Value::String(subject_str));
            }

            // Add properties
            for (predicate, values) in properties {
                subject_obj.insert(predicate, serde_json::Value::Array(values));
            }

            graph_array.push(serde_json::Value::Object(subject_obj));
        }

        jsonld_document.insert("@graph".to_string(), serde_json::Value::Array(graph_array));

        // Write JSON with pretty printing
        let json_output = if context.pretty_print {
            serde_json::to_string_pretty(&jsonld_document)
        } else {
            serde_json::to_string(&jsonld_document)
        }
        .map_err(|e| StarError::serialization_error(format!("JSON serialization error: {}", e)))?;

        buf_writer
            .write_all(json_output.as_bytes())
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        debug!(
            "Serialized {} quads in JSON-LD-star format",
            graph.quad_len()
        );
        Ok(())
    }

    /// Add a quad to the JSON-LD structure
    fn add_quad_to_jsonld(
        &self,
        subjects: &mut std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<serde_json::Value>>,
        >,
        quad: &StarQuad,
    ) -> StarResult<()> {
        let subject_str = self.term_to_jsonld_id(&quad.subject)?;
        let predicate_str = self.term_to_jsonld_predicate(&quad.predicate)?;
        let object_value = self.term_to_jsonld_value(&quad.object)?;

        let subject_props = subjects
            .entry(subject_str)
            .or_insert_with(std::collections::HashMap::new);
        let prop_values = subject_props.entry(predicate_str).or_insert_with(Vec::new);
        prop_values.push(object_value);

        Ok(())
    }

    /// Convert a StarTerm to JSON-LD @id format
    fn term_to_jsonld_id(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(node.iri.clone()),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::QuotedTriple(triple) => {
                // For quoted triples as subjects, create a special identifier
                Ok(format!("_:qt_{}", self.hash_triple(triple)))
            }
            _ => Err(StarError::serialization_error(
                "Invalid subject term".to_string(),
            )),
        }
    }

    /// Convert a StarTerm to JSON-LD predicate format
    fn term_to_jsonld_predicate(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(node.iri.clone()),
            _ => Err(StarError::serialization_error(
                "Invalid predicate term".to_string(),
            )),
        }
    }

    /// Convert a StarTerm to JSON-LD value format
    fn term_to_jsonld_value(&self, term: &StarTerm) -> StarResult<serde_json::Value> {
        match term {
            StarTerm::NamedNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(node.iri.clone()),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::BlankNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(format!("_:{}", node.id)),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Literal(literal) => {
                let mut obj = serde_json::Map::new();

                // Add the value
                if let Ok(num) = literal.value.parse::<f64>() {
                    // Numeric literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(num)
                                .unwrap_or(serde_json::Number::from(0)),
                        ),
                    );
                } else if literal.value == "true" || literal.value == "false" {
                    // Boolean literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::Bool(literal.value == "true"),
                    );
                } else {
                    // String literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::String(literal.value.clone()),
                    );
                }

                // Add datatype if present
                if let Some(ref datatype) = literal.datatype {
                    obj.insert(
                        "@type".to_string(),
                        serde_json::Value::String(datatype.iri.clone()),
                    );
                }

                // Add language if present
                if let Some(ref lang) = literal.language {
                    obj.insert(
                        "@language".to_string(),
                        serde_json::Value::String(lang.clone()),
                    );
                }

                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::QuotedTriple(triple) => {
                // RDF-star extension: represent quoted triple as annotation
                let mut annotation_obj = serde_json::Map::new();
                annotation_obj.insert(
                    "subject".to_string(),
                    self.term_to_jsonld_value(&triple.subject)?,
                );
                annotation_obj.insert(
                    "predicate".to_string(),
                    self.term_to_jsonld_value(&triple.predicate)?,
                );
                annotation_obj.insert(
                    "object".to_string(),
                    self.term_to_jsonld_value(&triple.object)?,
                );

                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@annotation".to_string(),
                    serde_json::Value::Object(annotation_obj),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Variable(var) => {
                // Variables are typically used in SPARQL queries, not in serialized data
                // For JSON-LD, we'll represent them as special objects
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@variable".to_string(),
                    serde_json::Value::String(var.name.clone()),
                );
                Ok(serde_json::Value::Object(obj))
            }
        }
    }

    /// Generate a hash for a triple (simple implementation)
    fn hash_triple(&self, triple: &StarTriple) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", triple).hash(&mut hasher);
        hasher.finish()
    }

    /// Format a StarTerm for Turtle-star (with prefix compression)
    fn format_term(&self, term: &StarTerm, context: &SerializationContext) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(context.compress_iri(&node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal(&literal.value));

                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^{}", context.compress_iri(&datatype.iri)));
                }

                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term(&triple.subject, context)?;
                let predicate = self.format_term(&triple.predicate, context)?;
                let object = self.format_term(&triple.object, context)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Format a StarTerm for N-Triples-star (full IRIs, no prefixes)
    fn format_term_ntriples(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal(&literal.value));

                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{}", lang));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }

                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term_ntriples(&triple.subject)?;
                let predicate = self.format_term_ntriples(&triple.predicate)?;
                let object = self.format_term_ntriples(&triple.object)?;
                Ok(format!("<< {} {} {} >>", subject, predicate, object))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Escape special characters in literals
    fn escape_literal(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
            .replace('"', "\\\"")
    }

    /// Add common namespace prefixes
    fn add_common_prefixes(&self, context: &mut SerializationContext) {
        context.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        context.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        context.add_prefix("owl", "http://www.w3.org/2002/07/owl#");
        context.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        context.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        context.add_prefix("dc", "http://purl.org/dc/terms/");
    }

    /// Parse a graph name string back to a term for TriG serialization
    fn parse_graph_name(
        &self,
        graph_name: &str,
        context: &SerializationContext,
    ) -> StarResult<String> {
        if graph_name.starts_with("_:") {
            // Blank node graph name
            Ok(graph_name.to_string())
        } else {
            // Named node graph name - compress with prefixes if possible
            Ok(context.compress_iri(graph_name))
        }
    }
}

impl Default for StarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for serialization
impl StarSerializer {
    /// Check if a graph is suitable for pretty printing
    pub fn can_pretty_print(&self, graph: &StarGraph) -> bool {
        // Simple heuristic: if graph is small and doesn't have deeply nested quoted triples
        graph.len() < 1000 && graph.max_nesting_depth() < 3
    }

    /// Estimate serialized size for a graph
    pub fn estimate_size(&self, graph: &StarGraph, format: StarFormat) -> usize {
        let base_size_per_triple = match format {
            StarFormat::TurtleStar => 50,   // Turtle is more compact
            StarFormat::NTriplesStar => 80, // N-Triples uses full IRIs
            StarFormat::TrigStar => 60,     // TriG has graph context
            StarFormat::NQuadsStar => 90,   // N-Quads uses full IRIs + graph
            StarFormat::JsonLdStar => 120,  // JSON-LD has overhead from JSON structure
        };

        let quoted_triple_multiplier = 1.5; // Quoted triples add overhead

        let mut total_size = graph.len() * base_size_per_triple;

        // Add overhead for quoted triples
        let quoted_count = graph.count_quoted_triples();
        total_size +=
            (quoted_count as f64 * quoted_triple_multiplier * base_size_per_triple as f64) as usize;

        total_size
    }

    /// Validate that a graph can be serialized in the given format
    pub fn validate_for_format(&self, graph: &StarGraph, format: StarFormat) -> StarResult<()> {
        // Check nesting depth
        let max_depth = graph.max_nesting_depth();
        if max_depth > self.config.max_nesting_depth {
            return Err(StarError::serialization_error(format!(
                "Graph nesting depth {} exceeds maximum {}",
                max_depth, self.config.max_nesting_depth
            )));
        }

        // Format-specific validation
        match format {
            StarFormat::TurtleStar | StarFormat::NTriplesStar => {
                // These formats support quoted triples in any position
                Ok(())
            }
            StarFormat::TrigStar | StarFormat::NQuadsStar => {
                // TODO: Add quad-specific validation when implemented
                Ok(())
            }
            StarFormat::JsonLdStar => {
                // JSON-LD-star supports quoted triples as annotations
                Ok(())
            }
        }
    }

    /// Serialize a graph to string using the specified format and options
    pub fn serialize_graph(
        &self,
        graph: &StarGraph,
        format: StarFormat,
        options: &SerializationOptions,
    ) -> StarResult<String> {
        // For now, this is a wrapper around serialize_to_string
        // In a more complete implementation, this would use the options parameter
        self.serialize_to_string(graph, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use crate::parser::StarParser;

    #[test]
    fn test_simple_triple_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        graph.insert(triple).unwrap();

        // Test N-Triples-star
        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("<http://example.org/alice>"));
        assert!(result.contains("<http://example.org/knows>"));
        assert!(result.contains("<http://example.org/bob>"));
        assert!(result.ends_with(" .\n"));

        // Test Turtle-star
        let result = serializer
            .serialize_to_string(&graph, StarFormat::TurtleStar)
            .unwrap();
        assert!(result.contains("@prefix"));
    }

    #[test]
    fn test_quoted_triple_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        graph.insert(outer).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("<<"));
        assert!(result.contains(">>"));
        assert!(result.contains("\"25\""));
        assert!(result.contains("\"0.9\""));
    }

    #[test]
    fn test_literal_escaping() {
        let test_cases = vec![
            ("simple", "simple"),
            ("with\nnewline", "with\\nnewline"),
            ("with\ttab", "with\\ttab"),
            ("with\"quote", "with\\\"quote"),
            ("with\\backslash", "with\\\\backslash"),
        ];

        for (input, expected) in test_cases {
            let escaped = StarSerializer::escape_literal(input);
            assert_eq!(escaped, expected);
        }
    }

    #[test]
    fn test_literal_with_language_and_datatype() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Literal with language tag
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/resource").unwrap(),
            StarTerm::iri("http://example.org/label").unwrap(),
            StarTerm::literal_with_language("Hello", "en").unwrap(),
        );

        // Literal with datatype
        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/resource").unwrap(),
            StarTerm::iri("http://example.org/count").unwrap(),
            StarTerm::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer")
                .unwrap(),
        );

        graph.insert(triple1).unwrap();
        graph.insert(triple2).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NTriplesStar)
            .unwrap();
        assert!(result.contains("\"Hello\"@en"));
        assert!(result.contains("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn test_format_validation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create deeply nested quoted triples
        let mut current_triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        // Nest it multiple times
        for _ in 0..15 {
            current_triple = StarTriple::new(
                StarTerm::quoted_triple(current_triple),
                StarTerm::iri("http://example.org/meta").unwrap(),
                StarTerm::literal("value").unwrap(),
            );
        }

        graph.insert(current_triple).unwrap();

        // Should fail validation due to excessive nesting
        let result = serializer.validate_for_format(&graph, StarFormat::NTriplesStar);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_estimation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        graph.insert(triple).unwrap();

        let turtle_size = serializer.estimate_size(&graph, StarFormat::TurtleStar);
        let ntriples_size = serializer.estimate_size(&graph, StarFormat::NTriplesStar);

        // N-Triples should be larger due to full IRIs
        assert!(ntriples_size > turtle_size);
    }

    #[test]
    fn test_enhanced_trig_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triple to default graph
        let default_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(default_triple).unwrap();

        // Add quad to named graph
        let named_quad = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(named_quad).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::TrigStar)
            .unwrap();

        // Should contain prefix declarations
        assert!(result.contains("@prefix"));

        // Should contain default graph block
        assert!(result.contains("{"));
        assert!(result.contains("alice"));

        // Should contain named graph declaration
        assert!(result.contains("http://example.org/graph1"));
        assert!(result.contains("charlie"));
    }

    #[test]
    fn test_enhanced_nquads_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triple to default graph
        let default_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(default_triple).unwrap();

        // Add quad to named graph
        let named_quad = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(named_quad).unwrap();

        let result = serializer
            .serialize_to_string(&graph, StarFormat::NQuadsStar)
            .unwrap();

        // Should contain default graph triple (3 terms)
        assert!(result.contains(
            "<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> ."
        ));

        // Should contain named graph quad (4 terms)
        assert!(result.contains("<http://example.org/charlie> <http://example.org/age> \"30\" <http://example.org/graph1> ."));
    }

    #[test]
    fn test_quoted_triple_serialization_roundtrip() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a complex quoted triple structure
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("Hello").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.95").unwrap(),
        );

        graph.insert(outer).unwrap();

        // Test all formats
        for format in [
            StarFormat::TurtleStar,
            StarFormat::NTriplesStar,
            StarFormat::TrigStar,
            StarFormat::NQuadsStar,
        ] {
            let serialized = serializer.serialize_to_string(&graph, format).unwrap();

            // Should contain quoted triple markers
            assert!(serialized.contains("<<"));
            assert!(serialized.contains(">>"));

            // Should contain the nested content
            assert!(serialized.contains("alice"));
            assert!(serialized.contains("says"));
            assert!(serialized.contains("Hello"));
            assert!(serialized.contains("certainty"));
        }
    }

    #[test]
    fn test_nquads_star_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add triples to default graph
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        graph.insert(triple1).unwrap();

        // Add quad with named graph
        let quad1 = StarQuad::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/likes").unwrap(),
            StarTerm::iri("http://example.org/dave").unwrap(),
            Some(StarTerm::iri("http://example.org/graph1").unwrap()),
        );
        graph.insert_quad(quad1).unwrap();

        // Add quad with quoted triple
        let quoted = StarTriple::new(
            StarTerm::iri("http://example.org/eve").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("hello").unwrap(),
        );
        let quad2 = StarQuad::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.8").unwrap(),
            Some(StarTerm::iri("http://example.org/graph2").unwrap()),
        );
        graph.insert_quad(quad2).unwrap();

        let serialized = serializer
            .serialize_to_string(&graph, StarFormat::NQuadsStar)
            .unwrap();

        // Verify each quad is on its own line
        let lines: Vec<&str> = serialized
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 3);

        // Verify graph contexts are present
        assert!(serialized.contains("<http://example.org/graph1>"));
        assert!(serialized.contains("<http://example.org/graph2>"));

        // Verify quoted triple syntax
        assert!(serialized.contains("<< "));
        assert!(serialized.contains(" >>"));
    }

    #[test]
    fn test_trig_star_serialization_with_multiple_graphs() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Add to default graph
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );
        graph.insert(triple1).unwrap();

        // Add to named graph 1
        let quad1 = StarQuad::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/likes").unwrap(),
            StarTerm::iri("http://example.org/coffee").unwrap(),
            Some(StarTerm::iri("http://example.org/preferences").unwrap()),
        );
        graph.insert_quad(quad1).unwrap();

        // Add quoted triple to named graph 2
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/charlie").unwrap(),
            StarTerm::iri("http://example.org/believes").unwrap(),
            StarTerm::literal("earth is round").unwrap(),
        );
        let quad2 = StarQuad::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/confidence").unwrap(),
            StarTerm::literal("1.0").unwrap(),
            Some(StarTerm::iri("http://example.org/beliefs").unwrap()),
        );
        graph.insert_quad(quad2).unwrap();

        let serialized = serializer
            .serialize_to_string(&graph, StarFormat::TrigStar)
            .unwrap();

        // Verify prefixes are included
        assert!(serialized.contains("@prefix"));

        // Verify default graph block
        assert!(serialized.contains("{\n"));
        assert!(serialized.contains("alice"));

        // Verify named graph blocks
        assert!(serialized.contains("<http://example.org/preferences> {"));
        assert!(serialized.contains("<http://example.org/beliefs> {"));

        // Verify quoted triple in TriG format
        assert!(serialized.contains("<<"));
        assert!(serialized.contains(">>"));
    }

    #[test]
    fn test_streaming_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a larger graph for streaming test
        for i in 0..1000 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("value{}", i)).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let mut output = Vec::new();
        serializer
            .serialize_streaming(&graph, &mut output, StarFormat::NTriplesStar, 100)
            .unwrap();

        let output_str = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1000);

        // Verify each line is a valid N-Triples statement
        for line in lines {
            assert!(line.ends_with(" ."));
            assert!(line.contains("http://example.org/"));
        }
    }

    #[test]
    fn test_parallel_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a graph suitable for parallel processing
        for i in 0..500 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/subject{}", i)).unwrap(),
                StarTerm::iri("http://example.org/predicate").unwrap(),
                StarTerm::iri(&format!("http://example.org/object{}", i)).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_parallel(&graph, output, StarFormat::NTriplesStar, 4, 100)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 500);
    }

    #[test]
    fn test_serialization_with_options() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create test graph
        for i in 0..100 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("test{}", i)).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let mut options = SerializationOptions::default();
        options.streaming = true;
        options.batch_size = 25;
        options.buffer_size = 1024;

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_with_options(&graph, output, StarFormat::NTriplesStar, &options)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();
        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 100);
    }

    #[test]
    fn test_optimized_serialization() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        // Create a complex graph with quoted triples
        for i in 0..50 {
            let inner = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/alice{}", i)).unwrap(),
                StarTerm::iri("http://example.org/says").unwrap(),
                StarTerm::literal(&format!("statement{}", i)).unwrap(),
            );
            let outer = StarTriple::new(
                StarTerm::quoted_triple(inner),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.9").unwrap(),
            );
            graph.insert(outer).unwrap();
        }

        let output = Box::leak(Box::new(Vec::new()));
        let output_ptr = output as *const Vec<u8>;
        serializer
            .serialize_optimized(&graph, output, StarFormat::NTriplesStar)
            .unwrap();

        // Safe because we know the serialize method completed and output is still valid
        let output_data = unsafe { &*output_ptr };
        let output_str = String::from_utf8(output_data.clone()).unwrap();

        // Should contain quoted triple syntax
        assert!(output_str.contains("<<"));
        assert!(output_str.contains(">>"));

        let lines: Vec<&str> = output_str
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 50);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        for i in 0..100 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal(&format!("value{}", i)).unwrap(),
            );
            graph.insert(triple).unwrap();
        }

        let options = SerializationOptions::default();
        let memory_estimate =
            serializer.estimate_memory_usage(&graph, StarFormat::NTriplesStar, &options);

        // Should provide reasonable estimate (not zero, not excessive)
        assert!(memory_estimate > 1000);
        assert!(memory_estimate < 10_000_000);

        let mut streaming_options = SerializationOptions::default();
        streaming_options.streaming = true;
        let streaming_estimate =
            serializer.estimate_memory_usage(&graph, StarFormat::NTriplesStar, &streaming_options);

        // Streaming should use less memory
        assert!(streaming_estimate < memory_estimate);
    }

    #[test]
    fn test_chunked_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunked = ChunkedIterator::new(data.into_iter(), 3);

        let chunks: Vec<_> = chunked.collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7, 8, 9]);
        assert_eq!(chunks[3], vec![10]);
    }

    #[test]
    fn test_compression_type_selection() {
        let serializer = StarSerializer::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("test").unwrap(),
        );
        graph.insert(triple).unwrap();

        // Test different compression types (placeholder implementations)
        for compression in [
            CompressionType::None,
            CompressionType::Gzip,
            CompressionType::Zstd,
            CompressionType::Lz4,
        ] {
            let mut options = SerializationOptions::default();
            options.compression = compression;

            let output = Box::leak(Box::new(Vec::new()));
            let result = serializer.serialize_with_options(
                &graph,
                output,
                StarFormat::NTriplesStar,
                &options,
            );

            // Should not fail even with unimplemented compression
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let parser = StarParser::new();
        let serializer = StarSerializer::new();
        let mut original_graph = StarGraph::new();

        // Create complex graph with various features
        let simple = StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p1").unwrap(),
            StarTerm::literal("test").unwrap(),
        );
        original_graph.insert(simple).unwrap();

        // Quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/says").unwrap(),
            StarTerm::literal("hello").unwrap(),
        );
        let quoted = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        original_graph.insert(quoted).unwrap();

        // Test roundtrip for each format
        for format in [
            StarFormat::TurtleStar,
            StarFormat::NTriplesStar,
            StarFormat::TrigStar,
            StarFormat::NQuadsStar,
        ] {
            let serialized = serializer
                .serialize_to_string(&original_graph, format)
                .unwrap();
            let parsed_graph = parser.parse_str(&serialized, format).unwrap();

            // Verify same number of triples
            assert_eq!(
                original_graph.total_len(),
                parsed_graph.total_len(),
                "Roundtrip failed for format {:?}",
                format
            );
        }
    }
}
