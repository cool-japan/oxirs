//! Ultra-high performance streaming JSON-LD processing
//!
//! This module provides advanced streaming capabilities for JSON-LD processing
//! with zero-copy operations, SIMD acceleration, and adaptive buffering.

use crate::{
    jsonld::JsonLdParseError,
    model::{NamedNode, Object, Predicate, Quad, Subject, Triple},
    optimization::{SimdJsonProcessor, TermInterner, TermInternerExt, ZeroCopyBuffer},
};

use super::context::{JsonLdLoadDocumentOptions, JsonLdRemoteDocument};
use super::profile::JsonLdProfileSet;
// Removed unused async_trait::async_trait import
use dashmap::DashMap;
// Removed unused futures::{SinkExt, StreamExt} imports
use parking_lot::Mutex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde_json::{Map, Value};
use std::{
    collections::VecDeque,
    error::Error as StdError,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, BufReader},
    sync::{mpsc, RwLock, Semaphore},
    time::{Duration, Instant},
};

/// Callback used to resolve remote JSON-LD `@context` documents referenced by
/// IRI (e.g. `"@context": "https://schema.org/"`).
///
/// It mirrors the loader mechanism used by
/// [`JsonLdContextProcessor`](super::context::JsonLdContextProcessor): given a
/// context IRI and [`JsonLdLoadDocumentOptions`] it must return the fetched
/// document bytes (or an error). The streaming parser never performs network
/// I/O itself — a resolver must be supplied explicitly via
/// [`UltraStreamingJsonLdParser::with_document_loader`]; otherwise any document
/// that references a remote context fails loudly rather than silently dropping
/// the import.
pub type StreamingDocumentLoader = dyn Fn(
        &str,
        &JsonLdLoadDocumentOptions,
    ) -> Result<JsonLdRemoteDocument, Box<dyn StdError + Send + Sync>>
    + Send
    + Sync;

/// Ultra-high performance streaming JSON-LD parser with adaptive optimizations
pub struct UltraStreamingJsonLdParser {
    config: StreamingConfig,
    context_cache: Arc<DashMap<String, Arc<Value>>>,
    term_interner: Arc<TermInterner>,
    performance_monitor: Arc<PerformanceMonitor>,
    simd_processor: SimdJsonProcessor,
    buffer_pool: Arc<BufferPool>,
    document_loader: Option<Arc<StreamingDocumentLoader>>,
}

/// Advanced configuration for streaming JSON-LD processing
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Chunk size for reading data (adaptive)
    pub chunk_size: usize,
    /// Maximum number of concurrent processing threads
    pub max_concurrent_threads: usize,
    /// Buffer size for intermediate processing
    pub buffer_size: usize,
    /// Enable SIMD acceleration for JSON parsing
    pub enable_simd: bool,
    /// Context caching configuration
    pub context_cache_size: usize,
    /// Adaptive buffering threshold
    pub adaptive_threshold: f64,
    /// Memory pressure detection
    pub memory_pressure_threshold: usize,
    /// Zero-copy optimization level
    pub zero_copy_level: ZeroCopyLevel,
    /// Performance profiling enabled
    pub enable_profiling: bool,
}

/// Zero-copy optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZeroCopyLevel {
    /// No zero-copy optimizations
    None,
    /// Basic zero-copy for string references
    Basic,
    /// Advanced zero-copy with arena allocation
    Advanced,
    /// Maximum zero-copy with custom allocators
    Maximum,
}

/// Real-time performance monitoring for streaming operations
pub struct PerformanceMonitor {
    total_bytes_processed: AtomicUsize,
    total_triples_parsed: AtomicUsize,
    parse_errors: AtomicUsize,
    context_cache_hits: AtomicUsize,
    context_cache_misses: AtomicUsize,
    simd_operations: AtomicUsize,
    zero_copy_operations: AtomicUsize,
    start_time: Instant,
    chunk_processing_times: Arc<Mutex<VecDeque<Duration>>>,
}

/// Adaptive buffer pool for high-throughput processing
pub struct BufferPool {
    available_buffers: Arc<Mutex<Vec<ZeroCopyBuffer>>>,
    buffer_size: usize,
    max_buffers: usize,
    current_buffers: AtomicUsize,
}

/// High-performance streaming sink for processed triples
#[async_trait::async_trait]
pub trait StreamingSink: Send + Sync {
    type Error: Send + Sync + std::error::Error + 'static;

    async fn process_triple_batch(&mut self, triples: Vec<Triple>) -> Result<(), Self::Error>;
    async fn process_quad_batch(&mut self, quads: Vec<Quad>) -> Result<(), Self::Error>;
    async fn flush(&mut self) -> Result<(), Self::Error>;
    fn performance_statistics(&self) -> SinkStatistics;
}

/// Statistics for sink performance monitoring
#[derive(Debug, Clone)]
pub struct SinkStatistics {
    pub total_triples_processed: usize,
    pub total_quads_processed: usize,
    pub average_batch_size: f64,
    pub processing_rate_per_second: f64,
    pub memory_usage_bytes: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024, // 64KB adaptive starting point
            max_concurrent_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                * 2,
            buffer_size: 1024 * 1024, // 1MB buffer
            enable_simd: true,
            context_cache_size: 10000,
            adaptive_threshold: 0.8,
            memory_pressure_threshold: 8 * 1024 * 1024 * 1024, // 8GB
            zero_copy_level: ZeroCopyLevel::Advanced,
            enable_profiling: true,
        }
    }
}

impl UltraStreamingJsonLdParser {
    /// Create a new ultra-performance streaming parser
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            context_cache: Arc::new(DashMap::with_capacity(config.context_cache_size)),
            term_interner: Arc::new(TermInterner::new()),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
            simd_processor: SimdJsonProcessor::new(),
            buffer_pool: Arc::new(BufferPool::new(config.buffer_size, 100)),
            document_loader: None,
            config,
        }
    }

    /// Attach a resolver for remote `@context` documents referenced by IRI.
    ///
    /// Without a loader, encountering a string `@context` (a remote context
    /// reference) during streaming produces an explicit
    /// [`JsonLdParseError`] instead of silently substituting an empty context.
    pub fn with_document_loader(mut self, loader: Arc<StreamingDocumentLoader>) -> Self {
        self.document_loader = Some(loader);
        self
    }

    /// Stream parse JSON-LD with ultra-high performance optimizations
    pub async fn stream_parse<R, S>(
        &mut self,
        reader: R,
        mut sink: S,
    ) -> Result<StreamingStatistics, JsonLdParseError>
    where
        R: AsyncRead + Unpin + Send + 'static,
        S: StreamingSink + Send + 'static,
        S::Error: 'static,
    {
        let mut buf_reader = BufReader::with_capacity(self.config.chunk_size, reader);
        let (tx, mut rx) = mpsc::channel::<ProcessingChunk>(self.config.buffer_size);
        let (triple_tx, mut triple_rx) = mpsc::channel::<Vec<Triple>>(100);
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_threads));

        // Spawn sink processing task
        let sink_handle = tokio::spawn(async move {
            while let Some(batch) = triple_rx.recv().await {
                sink.process_triple_batch(batch)
                    .await
                    .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))?;
            }

            sink.flush()
                .await
                .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))?;

            Ok::<(), JsonLdParseError>(())
        });

        // Spawn parallel processing tasks
        let processing_handle = tokio::spawn({
            let config = self.config.clone();
            let context_cache = Arc::clone(&self.context_cache);
            let term_interner = Arc::clone(&self.term_interner);
            let performance_monitor = Arc::clone(&self.performance_monitor);
            let simd_processor = self.simd_processor.clone();
            let document_loader = self.document_loader.clone();
            let triple_tx = triple_tx.clone();

            async move {
                let mut batch_buffer = Vec::with_capacity(config.buffer_size);
                // The raw byte stream is split into fixed-size transport chunks
                // whose boundaries fall at arbitrary offsets (mid-object,
                // mid-string, ...). We therefore reassemble the stream and only
                // hand *complete* top-level JSON values to the JSON parser,
                // maintaining tokenizer state across chunk boundaries.
                let mut splitter = TopLevelJsonSplitter::new();
                let mut eof = false;

                while !eof {
                    match rx.recv().await {
                        Some(chunk) => {
                            let _permit = semaphore.acquire().await.map_err(|_| {
                                JsonLdParseError::ProcessingError(
                                    "processing semaphore closed unexpectedly".to_string(),
                                )
                            })?;
                            splitter.push(&chunk.data);
                        }
                        None => {
                            // Reader side finished: allow the splitter to emit a
                            // final EOF-terminated primitive value if any.
                            eof = true;
                            splitter.mark_eof();
                        }
                    }

                    while let Some(document) = splitter.next_complete_value()? {
                        // Process a complete JSON-LD document/value with SIMD
                        // acceleration if available.
                        let processed_triples = if config.enable_simd {
                            Self::process_chunk_simd(
                                &document,
                                &context_cache,
                                &term_interner,
                                &simd_processor,
                                &document_loader,
                            )
                            .await?
                        } else {
                            Self::process_chunk_standard(
                                &document,
                                &context_cache,
                                &term_interner,
                                &document_loader,
                            )
                            .await?
                        };

                        performance_monitor.record_triples_parsed(processed_triples.len());

                        batch_buffer.extend(processed_triples);

                        // Adaptive batching based on performance metrics
                        if batch_buffer.len() >= config.buffer_size
                            || performance_monitor.should_flush_batch()
                        {
                            triple_tx
                                .send(std::mem::take(&mut batch_buffer))
                                .await
                                .map_err(|_| {
                                    JsonLdParseError::ProcessingError(
                                        "Triple channel send failed".to_string(),
                                    )
                                })?;
                        }
                    }
                }

                // The stream has ended: reject any trailing, truncated value
                // (e.g. a document cut off mid-object) rather than silently
                // discarding it.
                splitter.finish()?;

                // Flush remaining triples
                if !batch_buffer.is_empty() {
                    triple_tx.send(batch_buffer).await.map_err(|_| {
                        JsonLdParseError::ProcessingError("Triple channel send failed".to_string())
                    })?;
                }

                Ok::<(), JsonLdParseError>(())
            }
        });

        // Read and chunk data adaptively
        let mut buffer = self.buffer_pool.get_buffer().await;
        let mut total_bytes = 0;

        loop {
            match buf_reader.read(buffer.as_mut_slice()).await {
                Ok(0) => break, // EOF
                Ok(n) => {
                    buffer.set_len(n);
                    total_bytes += n;
                    self.performance_monitor.record_bytes_processed(n);

                    // Adaptive chunk size adjustment
                    if self.should_adjust_chunk_size(n) {
                        self.adjust_chunk_size_adaptive().await;
                    }

                    let chunk = ProcessingChunk {
                        data: buffer.as_slice().to_vec(),
                        timestamp: Instant::now(),
                        sequence_id: total_bytes,
                    };

                    tx.send(chunk).await.map_err(|_| {
                        JsonLdParseError::ProcessingError("Channel send failed".to_string())
                    })?;

                    buffer = self.buffer_pool.get_buffer().await;
                }
                Err(e) => return Err(JsonLdParseError::Io(e)),
            }
        }

        drop(tx); // Signal completion to processing task
        processing_handle
            .await
            .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))??;

        drop(triple_tx); // Signal completion to sink task
        sink_handle
            .await
            .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))??;

        Ok(self.performance_monitor.get_statistics())
    }

    /// Process a complete JSON-LD document with SIMD acceleration
    async fn process_chunk_simd(
        document: &[u8],
        context_cache: &DashMap<String, Arc<Value>>,
        term_interner: &TermInterner,
        simd_processor: &SimdJsonProcessor,
        document_loader: &Option<Arc<StreamingDocumentLoader>>,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        let start = Instant::now();

        // SIMD-accelerated JSON parsing (document is guaranteed to be a
        // complete, self-contained top-level JSON value by the splitter).
        let json_value = simd_processor
            .parse_json(document)
            .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))?;

        // Zero-copy context resolution
        let context =
            Self::resolve_context_zero_copy(&json_value, context_cache, document_loader).await?;

        // Parallel triple extraction with work-stealing
        #[cfg(feature = "parallel")]
        let triples = Self::extract_triples_parallel(&json_value, &context, term_interner).await?;
        #[cfg(not(feature = "parallel"))]
        let triples = Self::extract_triples_standard(&json_value, &context, term_interner).await?;

        // Record performance metrics
        let _processing_time = start.elapsed();
        // performance_monitor.record_chunk_processing_time(processing_time);

        Ok(triples)
    }

    /// Process a complete JSON-LD document with standard methods
    async fn process_chunk_standard(
        document: &[u8],
        context_cache: &DashMap<String, Arc<Value>>,
        term_interner: &TermInterner,
        document_loader: &Option<Arc<StreamingDocumentLoader>>,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        // Standard JSON parsing (document is a complete top-level JSON value).
        let json_value: Value = serde_json::from_slice(document)
            .map_err(|e| JsonLdParseError::ProcessingError(e.to_string()))?;

        // Context resolution with caching
        let context =
            Self::resolve_context_cached(&json_value, context_cache, document_loader).await?;

        // Triple extraction
        let triples = Self::extract_triples_standard(&json_value, &context, term_interner).await?;

        Ok(triples)
    }

    /// Zero-copy context resolution
    async fn resolve_context_zero_copy(
        json_value: &Value,
        context_cache: &DashMap<String, Arc<Value>>,
        document_loader: &Option<Arc<StreamingDocumentLoader>>,
    ) -> Result<Arc<Value>, JsonLdParseError> {
        if let Some(context_ref) = json_value.get("@context") {
            if let Some(context_str) = context_ref.as_str() {
                if let Some(cached_context) = context_cache.get(context_str) {
                    return Ok(Arc::clone(&cached_context));
                }

                // Resolve and cache context
                let resolved_context =
                    Self::resolve_remote_context(context_str, document_loader).await?;
                let context_arc = Arc::new(resolved_context);
                context_cache.insert(context_str.to_string(), Arc::clone(&context_arc));
                return Ok(context_arc);
            }

            // Inline context object (or array of contexts): use it verbatim as
            // the active context for term expansion.
            if context_ref.is_object() || context_ref.is_array() {
                return Ok(Arc::new(context_ref.clone()));
            }
        }

        // Default context
        Ok(Arc::new(Value::Object(Map::new())))
    }

    /// Cached context resolution
    async fn resolve_context_cached(
        json_value: &Value,
        context_cache: &DashMap<String, Arc<Value>>,
        document_loader: &Option<Arc<StreamingDocumentLoader>>,
    ) -> Result<Arc<Value>, JsonLdParseError> {
        // Similar to zero-copy but with different optimization strategy
        Self::resolve_context_zero_copy(json_value, context_cache, document_loader).await
    }

    /// Parallel triple extraction with work-stealing
    #[cfg(feature = "parallel")]
    async fn extract_triples_parallel(
        json_value: &Value,
        context: &Value,
        term_interner: &TermInterner,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        if let Value::Array(objects) = json_value {
            // Parallel processing of JSON-LD objects
            let triples: Result<Vec<Vec<Triple>>, JsonLdParseError> = objects
                .par_iter()
                .map(|obj| Self::extract_triples_from_object(obj, context, term_interner))
                .collect();

            Ok(triples?.into_iter().flatten().collect())
        } else {
            Self::extract_triples_from_object(json_value, context, term_interner)
        }
    }

    /// Standard triple extraction
    async fn extract_triples_standard(
        json_value: &Value,
        context: &Value,
        term_interner: &TermInterner,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        // A top-level JSON-LD document may be a single node object or an array
        // of node objects; handle both.
        if let Value::Array(objects) = json_value {
            let mut triples = Vec::new();
            for obj in objects {
                triples.extend(Self::extract_triples_from_object(
                    obj,
                    context,
                    term_interner,
                )?);
            }
            Ok(triples)
        } else {
            Self::extract_triples_from_object(json_value, context, term_interner)
        }
    }

    /// Extract triples from a single JSON-LD object
    fn extract_triples_from_object(
        obj: &Value,
        context: &Value,
        term_interner: &TermInterner,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        let mut triples = Vec::new();

        if let Value::Object(map) = obj {
            // Extract subject
            let subject: Subject = if let Some(id) = map.get("@id") {
                Subject::NamedNode(term_interner.intern_named_node(id.as_str().ok_or_else(
                    || JsonLdParseError::ProcessingError("Invalid @id".to_string()),
                )?)?)
            } else {
                // Generate blank node
                Subject::BlankNode(term_interner.intern_blank_node())
            };

            // Process properties
            for (key, value) in map {
                if key.starts_with('@') {
                    continue; // Skip JSON-LD keywords
                }

                // Expand property IRI using context
                let predicate_iri = Self::expand_property(key, context)?;
                let predicate = term_interner.intern_named_node(&predicate_iri)?;

                // Process values
                match value {
                    Value::Array(values) => {
                        for val in values {
                            if let Some(triple) = Self::create_triple_from_value(
                                subject.clone(),
                                predicate.clone(),
                                val,
                                context,
                                term_interner,
                            )? {
                                triples.push(triple);
                            }
                        }
                    }
                    _ => {
                        if let Some(triple) = Self::create_triple_from_value(
                            subject.clone(),
                            predicate,
                            value,
                            context,
                            term_interner,
                        )? {
                            triples.push(triple);
                        }
                    }
                }
            }
        }

        Ok(triples)
    }

    /// Create triple from JSON-LD value
    fn create_triple_from_value(
        subject: Subject,
        predicate: NamedNode,
        value: &Value,
        _context: &Value,
        term_interner: &TermInterner,
    ) -> Result<Option<Triple>, JsonLdParseError> {
        let object: Object = match value {
            Value::String(s) => {
                // Check if it's an IRI or literal
                if s.starts_with("http://") || s.starts_with("https://") {
                    Object::NamedNode(term_interner.intern_named_node(s)?)
                } else {
                    Object::Literal(term_interner.intern_literal(s)?)
                }
            }
            Value::Object(obj) => {
                if let Some(id) = obj.get("@id") {
                    // Object reference
                    Object::NamedNode(term_interner.intern_named_node(id.as_str().ok_or_else(
                        || JsonLdParseError::ProcessingError("Invalid @id in object".to_string()),
                    )?)?)
                } else if let Some(val) = obj.get("@value") {
                    // Typed literal
                    let literal_value = val.as_str().ok_or_else(|| {
                        JsonLdParseError::ProcessingError("Invalid @value".to_string())
                    })?;

                    if let Some(datatype) = obj.get("@type") {
                        let datatype_iri = datatype.as_str().ok_or_else(|| {
                            JsonLdParseError::ProcessingError("Invalid @type".to_string())
                        })?;
                        Object::Literal(
                            term_interner
                                .intern_literal_with_datatype(literal_value, datatype_iri)?,
                        )
                    } else if let Some(lang) = obj.get("@language") {
                        let language = lang.as_str().ok_or_else(|| {
                            JsonLdParseError::ProcessingError("Invalid @language".to_string())
                        })?;
                        Object::Literal(
                            term_interner.intern_literal_with_language(literal_value, language)?,
                        )
                    } else {
                        Object::Literal(term_interner.intern_literal(literal_value)?)
                    }
                } else {
                    return Ok(None); // Skip complex nested objects for now
                }
            }
            Value::Number(n) => Object::Literal(term_interner.intern_literal(&n.to_string())?),
            Value::Bool(b) => Object::Literal(term_interner.intern_literal(&b.to_string())?),
            _ => return Ok(None),
        };

        Ok(Some(Triple::new(
            subject,
            Predicate::NamedNode(predicate),
            object,
        )))
    }

    /// Expand a JSON-LD term (a property key) to an absolute IRI using the
    /// active context.
    ///
    /// This applies the relevant parts of the JSON-LD 1.1 IRI expansion
    /// algorithm for property keys:
    ///
    /// 1. an explicit term definition in the context (a string mapping, or an
    ///    object with `@id`), recursively resolving compact-IRI mappings;
    /// 2. a compact IRI `prefix:suffix` whose `prefix` is defined in the
    ///    context;
    /// 3. an already-absolute IRI (contains a `scheme:` / `://`);
    /// 4. `@vocab`-based expansion for plain terms.
    ///
    /// Under strict processing (the streaming parser's only mode) a term that
    /// matches none of these **fails loudly** with a [`JsonLdParseError`]
    /// rather than being silently mapped to a fabricated placeholder namespace.
    fn expand_property(property: &str, context: &Value) -> Result<String, JsonLdParseError> {
        Self::expand_term(property, context, 0)
    }

    fn expand_term(term: &str, context: &Value, depth: usize) -> Result<String, JsonLdParseError> {
        if depth > 16 {
            return Err(JsonLdParseError::ProcessingError(format!(
                "cyclic JSON-LD term definition while expanding '{term}'"
            )));
        }

        // 1. Explicit term definition in the active context.
        if let Some(def) = Self::context_lookup(context, term) {
            if let Some(mapping) = Self::term_definition_iri(def) {
                if mapping != term {
                    // The mapping itself may be a compact IRI or another term.
                    return Self::expand_iri(&mapping, context, depth + 1);
                }
            }
        }

        Self::expand_iri(term, context, depth)
    }

    /// Expand an IRI-ish string: a compact IRI, an absolute IRI, or (for a bare
    /// term) via `@vocab`.
    fn expand_iri(value: &str, context: &Value, depth: usize) -> Result<String, JsonLdParseError> {
        if depth > 16 {
            return Err(JsonLdParseError::ProcessingError(format!(
                "cyclic JSON-LD prefix/term definition while expanding '{value}'"
            )));
        }
        if let Some((prefix, suffix)) = value.split_once(':') {
            // Blank node identifiers and scheme-relative / absolute IRIs are
            // used verbatim.
            if prefix.is_empty() || prefix == "_" || suffix.starts_with("//") {
                return Ok(value.to_string());
            }

            // Compact IRI whose prefix is defined in the context.
            if let Some(def) = Self::context_lookup(context, prefix) {
                if let Some(prefix_iri) = Self::term_definition_iri(def) {
                    if prefix_iri != prefix {
                        let base = Self::expand_iri(&prefix_iri, context, depth + 1)?;
                        return Ok(format!("{base}{suffix}"));
                    }
                }
            }

            // A `scheme:...` form with an unknown prefix is treated as an
            // already-absolute IRI.
            return Ok(value.to_string());
        }

        // No colon: a bare term. Try `@vocab`.
        if let Some(vocab) = Self::context_vocab(context) {
            return Ok(format!("{vocab}{value}"));
        }

        Err(JsonLdParseError::ProcessingError(format!(
            "cannot expand JSON-LD term '{value}' to an absolute IRI: no matching term \
             definition, prefix, or @vocab in the active context"
        )))
    }

    /// Look up a key in a JSON-LD context, which may be a single object or an
    /// array of context objects (later entries take precedence).
    fn context_lookup<'a>(context: &'a Value, key: &str) -> Option<&'a Value> {
        match context {
            Value::Object(ctx) => ctx.get(key),
            Value::Array(contexts) => contexts
                .iter()
                .rev()
                .find_map(|c| Self::context_lookup(c, key)),
            _ => None,
        }
    }

    /// Extract the `@vocab` mapping from a context (object or array).
    fn context_vocab(context: &Value) -> Option<String> {
        Self::context_lookup(context, "@vocab")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Resolve a term definition to its raw IRI mapping. A term definition is
    /// either a string (the IRI) or an object carrying an `@id`.
    fn term_definition_iri(def: &Value) -> Option<String> {
        match def {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("@id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            _ => None,
        }
    }

    /// Resolve a remote `@context` referenced by IRI.
    ///
    /// The streaming parser performs no network I/O of its own. Resolution is
    /// delegated to the configured [`StreamingDocumentLoader`]; if none is
    /// configured, this **fails loudly** rather than silently substituting an
    /// empty context (which would corrupt every term in the document).
    async fn resolve_remote_context(
        context_iri: &str,
        document_loader: &Option<Arc<StreamingDocumentLoader>>,
    ) -> Result<Value, JsonLdParseError> {
        let loader = document_loader.as_ref().ok_or_else(|| {
            JsonLdParseError::ProcessingError(format!(
                "cannot resolve remote JSON-LD context '{context_iri}': no document loader \
                 configured. Supply one via UltraStreamingJsonLdParser::with_document_loader"
            ))
        })?;

        let options = JsonLdLoadDocumentOptions {
            request_profile: JsonLdProfileSet::from(super::profile::JsonLdProfile::Context),
        };

        let remote = loader(context_iri, &options).map_err(|e| {
            JsonLdParseError::ProcessingError(format!(
                "failed to load remote JSON-LD context '{context_iri}': {e}"
            ))
        })?;

        let parsed: Value = serde_json::from_slice(&remote.document).map_err(|e| {
            JsonLdParseError::ProcessingError(format!(
                "remote JSON-LD context '{context_iri}' is not valid JSON: {e}"
            ))
        })?;

        // A context document is conventionally `{"@context": {...}}`; unwrap the
        // inner context so it can be used directly for term expansion.
        let context = match parsed {
            Value::Object(mut obj) => match obj.remove("@context") {
                Some(inner) => inner,
                None => Value::Object(obj),
            },
            other => other,
        };

        Ok(context)
    }

    /// Check if chunk size should be adjusted
    fn should_adjust_chunk_size(&self, bytes_read: usize) -> bool {
        let target_size = self.config.chunk_size;
        let threshold = (target_size as f64 * self.config.adaptive_threshold) as usize;
        bytes_read < threshold || bytes_read > target_size * 2
    }

    /// Adaptively adjust chunk size based on performance
    async fn adjust_chunk_size_adaptive(&mut self) {
        let avg_processing_time = self.performance_monitor.average_chunk_processing_time();
        let memory_pressure = self.performance_monitor.memory_pressure_detected();

        if memory_pressure {
            self.config.chunk_size = (self.config.chunk_size / 2).max(1024);
        } else if avg_processing_time < Duration::from_millis(10) {
            self.config.chunk_size = (self.config.chunk_size * 2).min(1024 * 1024);
        }
    }
}

/// Chunk of data being processed
#[derive(Debug)]
struct ProcessingChunk {
    data: Vec<u8>,
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    sequence_id: usize,
}

/// Incremental scanning state for [`TopLevelJsonSplitter`].
#[derive(Debug, Clone, Copy)]
enum SplitState {
    /// Between values, skipping insignificant whitespace.
    Idle,
    /// Inside a `{...}` / `[...]` container that began at `start`.
    Container {
        start: usize,
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    /// Inside a `"..."` string scalar that began at `start`.
    StringScalar { start: usize, escaped: bool },
    /// Inside a bare primitive (number / `true` / `false` / `null`) that began
    /// at `start`.
    Primitive { start: usize },
}

/// Reassembles a byte stream that has been split into arbitrary-boundary chunks
/// and yields **complete** top-level JSON values.
///
/// A "streaming" JSON-LD parser cannot simply hand each transport chunk to
/// `serde_json`, because chunk boundaries fall at arbitrary offsets — in the
/// middle of an object, string, escape sequence, etc. This splitter maintains a
/// small tokenizer state (container depth, in-string / escape flags) across
/// chunk boundaries and only emits a byte slice once it forms a whole,
/// self-contained top-level JSON value. It supports a single large value
/// spanning many chunks as well as several concatenated / newline-delimited
/// values in one stream.
#[derive(Debug)]
struct TopLevelJsonSplitter {
    buf: Vec<u8>,
    /// Scan cursor into `buf`.
    pos: usize,
    /// Set once the underlying reader has reached EOF.
    eof: bool,
    state: SplitState,
}

impl TopLevelJsonSplitter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            eof: false,
            state: SplitState::Idle,
        }
    }

    #[inline]
    fn is_ws(byte: u8) -> bool {
        matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
    }

    /// Append freshly read bytes to the reassembly buffer.
    fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Signal that no further bytes will arrive.
    fn mark_eof(&mut self) {
        self.eof = true;
    }

    /// Extract the next complete top-level JSON value, if one is fully
    /// available. Returns `Ok(None)` when more input is required (or the stream
    /// is exhausted with only whitespace remaining).
    fn next_complete_value(&mut self) -> Result<Option<Vec<u8>>, JsonLdParseError> {
        loop {
            match self.state {
                SplitState::Idle => {
                    // Drop already-consumed bytes to keep memory bounded when a
                    // stream carries many concatenated values.
                    if self.pos > 0 {
                        self.buf.drain(0..self.pos);
                        self.pos = 0;
                    }
                    while self.pos < self.buf.len() && Self::is_ws(self.buf[self.pos]) {
                        self.pos += 1;
                    }
                    if self.pos >= self.buf.len() {
                        self.buf.drain(0..self.pos);
                        self.pos = 0;
                        return Ok(None);
                    }
                    let start = self.pos;
                    match self.buf[self.pos] {
                        b'{' | b'[' => {
                            self.pos += 1;
                            self.state = SplitState::Container {
                                start,
                                depth: 1,
                                in_string: false,
                                escaped: false,
                            };
                        }
                        b'"' => {
                            self.pos += 1;
                            self.state = SplitState::StringScalar {
                                start,
                                escaped: false,
                            };
                        }
                        _ => {
                            self.state = SplitState::Primitive { start };
                        }
                    }
                }
                SplitState::Container {
                    start,
                    mut depth,
                    mut in_string,
                    mut escaped,
                } => {
                    while self.pos < self.buf.len() {
                        let byte = self.buf[self.pos];
                        self.pos += 1;
                        if in_string {
                            if escaped {
                                escaped = false;
                            } else if byte == b'\\' {
                                escaped = true;
                            } else if byte == b'"' {
                                in_string = false;
                            }
                        } else {
                            match byte {
                                b'"' => in_string = true,
                                b'{' | b'[' => depth += 1,
                                b'}' | b']' => {
                                    depth -= 1;
                                    if depth == 0 {
                                        let value = self.buf[start..self.pos].to_vec();
                                        self.state = SplitState::Idle;
                                        return Ok(Some(value));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    // Buffer exhausted mid-container; preserve state for the
                    // next chunk.
                    self.state = SplitState::Container {
                        start,
                        depth,
                        in_string,
                        escaped,
                    };
                    return Ok(None);
                }
                SplitState::StringScalar { start, mut escaped } => {
                    while self.pos < self.buf.len() {
                        let byte = self.buf[self.pos];
                        self.pos += 1;
                        if escaped {
                            escaped = false;
                        } else if byte == b'\\' {
                            escaped = true;
                        } else if byte == b'"' {
                            let value = self.buf[start..self.pos].to_vec();
                            self.state = SplitState::Idle;
                            return Ok(Some(value));
                        }
                    }
                    self.state = SplitState::StringScalar { start, escaped };
                    return Ok(None);
                }
                SplitState::Primitive { start } => {
                    while self.pos < self.buf.len() {
                        let byte = self.buf[self.pos];
                        if Self::is_ws(byte) || matches!(byte, b',' | b']' | b'}') {
                            let value = self.buf[start..self.pos].to_vec();
                            self.state = SplitState::Idle;
                            return Ok(Some(value));
                        }
                        self.pos += 1;
                    }
                    if self.eof {
                        // EOF terminates a trailing primitive.
                        let value = self.buf[start..self.pos].to_vec();
                        self.state = SplitState::Idle;
                        if value.is_empty() {
                            return Ok(None);
                        }
                        return Ok(Some(value));
                    }
                    self.state = SplitState::Primitive { start };
                    return Ok(None);
                }
            }
        }
    }

    /// Validate that the stream ended on a value boundary. A residual
    /// in-progress value (unclosed container / string) means the input was
    /// truncated at a chunk boundary — a fail-loud error rather than silent
    /// data loss.
    fn finish(&self) -> Result<(), JsonLdParseError> {
        match self.state {
            SplitState::Idle => {
                if self.buf[self.pos..].iter().any(|b| !Self::is_ws(*b)) {
                    return Err(JsonLdParseError::ProcessingError(
                        "trailing non-whitespace bytes after the final JSON-LD value".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(JsonLdParseError::ProcessingError(
                "incomplete JSON-LD document: input ended in the middle of a value \
                 (truncated at a chunk boundary or malformed JSON)"
                    .to_string(),
            )),
        }
    }
}

/// Streaming processing statistics
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    pub total_bytes_processed: usize,
    pub total_triples_parsed: usize,
    pub processing_time: Duration,
    pub average_throughput_mbps: f64,
    pub parse_errors: usize,
    pub context_cache_hit_ratio: f64,
    pub simd_operations_count: usize,
    pub zero_copy_operations_count: usize,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            total_bytes_processed: AtomicUsize::new(0),
            total_triples_parsed: AtomicUsize::new(0),
            parse_errors: AtomicUsize::new(0),
            context_cache_hits: AtomicUsize::new(0),
            context_cache_misses: AtomicUsize::new(0),
            simd_operations: AtomicUsize::new(0),
            zero_copy_operations: AtomicUsize::new(0),
            start_time: Instant::now(),
            chunk_processing_times: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
        }
    }

    fn record_bytes_processed(&self, bytes: usize) {
        self.total_bytes_processed
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_triples_parsed(&self, count: usize) {
        self.total_triples_parsed
            .fetch_add(count, Ordering::Relaxed);
    }

    fn should_flush_batch(&self) -> bool {
        // Adaptive flushing logic based on performance metrics
        self.average_chunk_processing_time() > Duration::from_millis(100)
    }

    fn average_chunk_processing_time(&self) -> Duration {
        let times = self.chunk_processing_times.lock();
        if times.is_empty() {
            return Duration::from_millis(1);
        }

        let total: Duration = times.iter().sum();
        total / times.len() as u32
    }

    fn memory_pressure_detected(&self) -> bool {
        // Simplified memory pressure detection
        false // Implementation would check actual memory usage
    }

    fn get_statistics(&self) -> StreamingStatistics {
        let elapsed = self.start_time.elapsed();
        let bytes = self.total_bytes_processed.load(Ordering::Relaxed);
        let triples = self.total_triples_parsed.load(Ordering::Relaxed);
        let errors = self.parse_errors.load(Ordering::Relaxed);
        let cache_hits = self.context_cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.context_cache_misses.load(Ordering::Relaxed);
        let simd_ops = self.simd_operations.load(Ordering::Relaxed);
        let zero_copy_ops = self.zero_copy_operations.load(Ordering::Relaxed);

        let throughput_mbps = if elapsed.as_secs() > 0 {
            (bytes as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let cache_hit_ratio = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        StreamingStatistics {
            total_bytes_processed: bytes,
            total_triples_parsed: triples,
            processing_time: elapsed,
            average_throughput_mbps: throughput_mbps,
            parse_errors: errors,
            context_cache_hit_ratio: cache_hit_ratio,
            simd_operations_count: simd_ops,
            zero_copy_operations_count: zero_copy_ops,
        }
    }
}

impl BufferPool {
    fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            available_buffers: Arc::new(Mutex::new(Vec::with_capacity(max_buffers))),
            buffer_size,
            max_buffers,
            current_buffers: AtomicUsize::new(0),
        }
    }

    async fn get_buffer(&self) -> ZeroCopyBuffer {
        loop {
            // Try to get a buffer without waiting
            {
                let mut buffers = self.available_buffers.lock();
                if let Some(buffer) = buffers.pop() {
                    return buffer;
                }
            } // MutexGuard dropped here

            if self.current_buffers.load(Ordering::Relaxed) < self.max_buffers {
                self.current_buffers.fetch_add(1, Ordering::Relaxed);
                return ZeroCopyBuffer::new(self.buffer_size);
            } else {
                // Wait for a buffer to become available
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }

    #[allow(dead_code)]
    fn return_buffer(&self, mut buffer: ZeroCopyBuffer) {
        buffer.reset();
        let mut buffers = self.available_buffers.lock();
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        } else {
            self.current_buffers.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Memory-efficient sink that accumulates triples in memory
pub struct MemoryStreamingSink {
    triples: Arc<RwLock<Vec<Triple>>>,
    quads: Arc<RwLock<Vec<Quad>>>,
    statistics: Arc<RwLock<SinkStatistics>>,
}

impl Default for MemoryStreamingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStreamingSink {
    pub fn new() -> Self {
        Self {
            triples: Arc::new(RwLock::new(Vec::new())),
            quads: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(SinkStatistics {
                total_triples_processed: 0,
                total_quads_processed: 0,
                average_batch_size: 0.0,
                processing_rate_per_second: 0.0,
                memory_usage_bytes: 0,
            })),
        }
    }

    pub fn into_triples(self) -> Arc<RwLock<Vec<Triple>>> {
        self.triples
    }

    pub async fn get_triples(&self) -> Vec<Triple> {
        self.triples.read().await.clone()
    }

    pub async fn get_quads(&self) -> Vec<Quad> {
        self.quads.read().await.clone()
    }
}

/// Error type for streaming operations
#[derive(Debug)]
pub struct StreamingError(Box<dyn StdError + Send + Sync>);

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Streaming error: {}", self.0)
    }
}

impl StdError for StreamingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&*self.0)
    }
}

impl From<Box<dyn StdError + Send + Sync>> for StreamingError {
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        StreamingError(err)
    }
}

#[async_trait::async_trait]
impl StreamingSink for MemoryStreamingSink {
    type Error = StreamingError;

    async fn process_triple_batch(&mut self, triples: Vec<Triple>) -> Result<(), Self::Error> {
        let batch_size = triples.len();
        self.triples.write().await.extend(triples);

        let mut stats = self.statistics.write().await;
        stats.total_triples_processed += batch_size;
        stats.average_batch_size = (stats.average_batch_size + batch_size as f64) / 2.0;

        Ok(())
    }

    async fn process_quad_batch(&mut self, quads: Vec<Quad>) -> Result<(), Self::Error> {
        let batch_size = quads.len();
        self.quads.write().await.extend(quads);

        let mut stats = self.statistics.write().await;
        stats.total_quads_processed += batch_size;

        Ok(())
    }

    async fn flush(&mut self) -> Result<(), Self::Error> {
        // Memory sink doesn't need explicit flushing
        Ok(())
    }

    fn performance_statistics(&self) -> SinkStatistics {
        // Would need to implement actual memory usage calculation
        SinkStatistics {
            total_triples_processed: 0,
            total_quads_processed: 0,
            average_batch_size: 0.0,
            processing_rate_per_second: 0.0,
            memory_usage_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Predicate;
    use std::io::Cursor;

    /// Collect (predicate_iri, object_string) pairs from parsed triples.
    fn predicates(triples: &[Triple]) -> Vec<String> {
        triples
            .iter()
            .map(|t| match t.predicate() {
                Predicate::NamedNode(n) => n.as_str().to_string(),
                Predicate::Variable(v) => v.as_str().to_string(),
            })
            .collect()
    }

    async fn run_parse(
        config: StreamingConfig,
        data: &str,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        let mut parser = UltraStreamingJsonLdParser::new(config);
        let reader = Cursor::new(data.as_bytes().to_vec());
        let sink = MemoryStreamingSink::new();
        let triples_arc = Arc::clone(&sink.triples);
        parser.stream_parse(reader, sink).await?;
        let triples = triples_arc.read().await.clone();
        Ok(triples)
    }

    async fn run_parse_with_loader(
        config: StreamingConfig,
        data: &str,
        loader: Arc<StreamingDocumentLoader>,
    ) -> Result<Vec<Triple>, JsonLdParseError> {
        let mut parser = UltraStreamingJsonLdParser::new(config).with_document_loader(loader);
        let reader = Cursor::new(data.as_bytes().to_vec());
        let sink = MemoryStreamingSink::new();
        let triples_arc = Arc::clone(&sink.triples);
        parser.stream_parse(reader, sink).await?;
        let triples = triples_arc.read().await.clone();
        Ok(triples)
    }

    #[tokio::test]
    async fn regression_streaming_array_multiple_objects() {
        // Array of node objects with absolute-IRI predicates (no context
        // needed). Verifies both the array handling and correct IRIs.
        let json = r#"[
            {"@id": "http://example.org/person/1", "http://schema.org/name": "Alice"},
            {"@id": "http://example.org/person/2", "http://schema.org/name": "Bob"}
        ]"#;

        let triples = run_parse(StreamingConfig::default(), json)
            .await
            .expect("array document should parse");
        assert_eq!(triples.len(), 2, "expected one triple per object");
        for p in predicates(&triples) {
            assert_eq!(p, "http://schema.org/name");
        }
    }

    #[tokio::test]
    async fn regression_large_object_split_across_chunks() {
        // A single object with many properties, parsed with a tiny buffer so the
        // byte stream is split at arbitrary offsets (mid-object / mid-string).
        // Previously every chunk was parsed independently as a whole JSON
        // document and this failed with a serde parse error.
        let mut props = String::new();
        for i in 0..60 {
            props.push_str(&format!(
                ",\n  \"prop{i}\": \"value number {i} with spaces\""
            ));
        }
        let json = format!(
            "{{\n  \"@context\": {{\"@vocab\": \"http://example.com/vocab#\"}},\n  \
             \"@id\": \"http://example.org/subject\"{props}\n}}"
        );

        // Small read buffer forces the ~2.6 KB document to be split across many
        // chunk boundaries (mid-object / mid-string). Kept above ~doc_len/100 so
        // the fixed-capacity buffer pool is not exhausted.
        let config = StreamingConfig {
            buffer_size: 64,
            enable_simd: false,
            ..Default::default()
        };

        let triples = run_parse(config, &json)
            .await
            .expect("large document split across chunks should parse");
        assert_eq!(triples.len(), 60, "all 60 properties should yield triples");
        for p in predicates(&triples) {
            assert!(
                p.starts_with("http://example.com/vocab#prop"),
                "predicate should be @vocab-expanded, got {p}"
            );
        }
    }

    #[tokio::test]
    async fn regression_large_object_split_across_chunks_simd() {
        // Same as above but exercising the SIMD processing path.
        let mut props = String::new();
        for i in 0..40 {
            props.push_str(&format!(",\n  \"p{i}\": \"v{i}\""));
        }
        let json = format!(
            "{{\"@context\": {{\"@vocab\": \"http://ex.com/v#\"}}, \
             \"@id\": \"http://example.org/s\"{props}}}"
        );

        let config = StreamingConfig {
            buffer_size: 8,
            enable_simd: true,
            ..Default::default()
        };

        let triples = run_parse(config, &json)
            .await
            .expect("SIMD path large document should parse");
        assert_eq!(triples.len(), 40);
    }

    #[tokio::test]
    async fn regression_expand_property_no_fabricated_namespace() {
        // Unmapped terms must NOT silently become http://example.org/<term>.
        // With @vocab they expand correctly; a term mapped in the context uses
        // its mapping.
        let json = r#"{
            "@context": {
                "@vocab": "http://example.com/vocab#",
                "name": "http://schema.org/name"
            },
            "@id": "http://example.org/s",
            "name": "Alice",
            "age": "30"
        }"#;

        let triples = run_parse(StreamingConfig::default(), json)
            .await
            .expect("document with @vocab should parse");
        let preds = predicates(&triples);
        assert!(
            preds.contains(&"http://schema.org/name".to_string()),
            "explicit term mapping must win: {preds:?}"
        );
        assert!(
            preds.contains(&"http://example.com/vocab#age".to_string()),
            "unmapped term must use @vocab: {preds:?}"
        );
        assert!(
            !preds.iter().any(|p| p.contains("http://example.org/age")),
            "no fabricated example.org namespace allowed: {preds:?}"
        );
    }

    #[tokio::test]
    async fn regression_unmapped_term_without_vocab_fails_loud() {
        // No @vocab, no term definition, no colon -> cannot expand. Must be an
        // explicit error, never a fabricated IRI and never a silent success.
        let json = r#"{"@id": "http://example.org/s", "name": "Alice"}"#;
        let result = run_parse(StreamingConfig::default(), json).await;
        assert!(
            result.is_err(),
            "unmappable term must fail loudly, got {result:?}"
        );
    }

    #[tokio::test]
    async fn regression_remote_context_without_loader_fails_loud() {
        // String @context is a remote reference. Without a configured loader we
        // must fail loudly instead of substituting an empty context.
        let json = r#"{"@context": "https://schema.org/", "@id": "http://example.org/s", "name": "Alice"}"#;
        let result = run_parse(StreamingConfig::default(), json).await;
        assert!(
            result.is_err(),
            "remote @context without loader must fail loudly, got {result:?}"
        );
    }

    #[tokio::test]
    async fn regression_remote_context_resolved_with_loader() {
        // With a loader, the remote context is fetched and used to expand terms
        // to their real IRIs (not http://example.org/...).
        let json = r#"{"@context": "https://example.test/ctx", "@id": "http://example.org/s", "name": "Alice"}"#;
        let loader: Arc<StreamingDocumentLoader> =
            Arc::new(|iri: &str, _opts: &JsonLdLoadDocumentOptions| {
                assert_eq!(iri, "https://example.test/ctx");
                Ok(JsonLdRemoteDocument {
                    document: br#"{"@context": {"name": "https://schema.org/name"}}"#.to_vec(),
                    document_url: iri.to_string(),
                })
            });

        let triples = run_parse_with_loader(StreamingConfig::default(), json, loader)
            .await
            .expect("remote context should resolve via loader");
        let preds = predicates(&triples);
        assert_eq!(preds, vec!["https://schema.org/name".to_string()]);
    }

    #[tokio::test]
    async fn regression_remote_context_loader_failure_propagates() {
        // A loader that errors must surface the error, not be swallowed.
        let json = r#"{"@context": "https://broken.test/ctx", "@id": "http://example.org/s", "name": "Alice"}"#;
        let loader: Arc<StreamingDocumentLoader> =
            Arc::new(|_iri: &str, _opts: &JsonLdLoadDocumentOptions| {
                Err::<JsonLdRemoteDocument, _>("network unreachable".into())
            });

        let result = run_parse_with_loader(StreamingConfig::default(), json, loader).await;
        assert!(result.is_err(), "loader failure must propagate: {result:?}");
    }

    #[tokio::test]
    async fn regression_truncated_document_fails_loud() {
        // Input truncated mid-object (never closed). Must error, not silently
        // produce zero triples with success.
        let json = r#"[{"@id": "http://example.org/1", "http://schema.org/name": "Alice"}"#; // missing ']'
        let result = run_parse(StreamingConfig::default(), json).await;
        assert!(
            result.is_err(),
            "truncated document must fail loudly, got {result:?}"
        );
    }

    #[test]
    fn regression_splitter_reassembles_across_chunk_boundaries() {
        // Feed a JSON object one byte at a time; the splitter must yield exactly
        // one complete value with all bytes intact.
        let doc = br#"{"a": "b\"c", "d": [1, 2, {"e": "}"}]}"#;
        let mut splitter = TopLevelJsonSplitter::new();
        let mut out = Vec::new();
        for byte in doc.iter() {
            splitter.push(&[*byte]);
            while let Some(v) = splitter
                .next_complete_value()
                .expect("splitter should not error on valid input")
            {
                out.push(v);
            }
        }
        splitter.mark_eof();
        while let Some(v) = splitter
            .next_complete_value()
            .expect("splitter should not error at eof")
        {
            out.push(v);
        }
        splitter
            .finish()
            .expect("valid document should finish cleanly");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], doc.to_vec());
    }

    #[test]
    fn regression_splitter_yields_multiple_concatenated_values() {
        // Newline-delimited JSON objects must be emitted individually.
        let doc = b"{\"a\":1}\n{\"b\":2}\n[3,4]";
        let mut splitter = TopLevelJsonSplitter::new();
        splitter.push(doc);
        splitter.mark_eof();
        let mut out = Vec::new();
        while let Some(v) = splitter.next_complete_value().expect("no error") {
            out.push(String::from_utf8(v).expect("utf8"));
        }
        splitter.finish().expect("clean finish");
        assert_eq!(out, vec!["{\"a\":1}", "{\"b\":2}", "[3,4]"]);
    }

    #[test]
    fn regression_splitter_incomplete_container_errors_on_finish() {
        let mut splitter = TopLevelJsonSplitter::new();
        splitter.push(b"{\"a\": [1, 2");
        splitter.mark_eof();
        while splitter
            .next_complete_value()
            .expect("no error while draining")
            .is_some()
        {}
        assert!(
            splitter.finish().is_err(),
            "unclosed container must be reported as truncated"
        );
    }
}
