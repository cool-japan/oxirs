//! Serialization configuration types and contexts

use std::collections::HashMap;

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
pub(crate) struct SerializationContext {
    /// Namespace prefixes for compact representation
    pub(crate) prefixes: HashMap<String, String>,
    /// Base IRI for relative references
    #[allow(dead_code)]
    pub(crate) base_iri: Option<String>,
    /// Pretty printing with indentation
    pub(crate) pretty_print: bool,
    /// Current indentation level
    pub(crate) indent_level: usize,
    /// Indentation string (spaces or tabs)
    pub(crate) indent_string: String,
}

impl SerializationContext {
    pub(crate) fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base_iri: None,
            pretty_print: true,
            indent_level: 0,
            indent_string: "  ".to_string(),
        }
    }

    /// Add a namespace prefix
    pub(crate) fn add_prefix(&mut self, prefix: &str, namespace: &str) {
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
    }

    /// Get current indentation
    pub(crate) fn current_indent(&self) -> String {
        self.indent_string.repeat(self.indent_level)
    }

    /// Increase indentation level
    pub(crate) fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    pub(crate) fn decrease_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// Try to compress IRI using prefixes
    pub(crate) fn compress_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                return format!("{prefix}:{local}");
            }
        }

        // Return full IRI if no prefix match
        format!("<{iri}>")
    }
}
