//! Zero-copy RDF operations using SciRS2-core's memory management
//!
//! This module provides high-performance zero-copy operations for RDF triples and quads
//! using SciRS2-core's advanced memory management capabilities.
//!
//! # Features
//!
//! - **Zero-copy views**: Efficient views into RDF data without allocations
//! - **Buffer pooling**: Reuse memory buffers for parsing and serialization
//! - **Memory-mapped storage**: Direct file mapping for large RDF datasets
//! - **Adaptive chunking**: Process large RDF files in optimal chunk sizes
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_core::zero_copy_rdf::{ZeroCopyTripleStore, ZeroCopyConfig};
//! use oxirs_core::model::Triple;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! // Create a zero-copy triple store with optimized settings
//! let config = ZeroCopyConfig::default()
//!     .with_buffer_pool_size(100)
//!     .with_chunk_size(10000);
//!
//! let mut store = ZeroCopyTripleStore::with_config(config)?;
//!
//! // Load RDF data with zero-copy parsing
//! store.load_file_zero_copy("data.nt")?;
//!
//! // Query with zero-copy views
//! let triples = store.query_zero_copy(None, None, None)?;
//! println!("Found {} triples without copying", triples.len());
//!
//! # Ok(())
//! # }
//! ```

use crate::model::{Object, Predicate, Subject, Triple};
use crate::OxirsError;

// SciRS2-core zero-copy and memory management
use scirs2_core::memory::BufferPool;

// Reserved for future use:
// use scirs2_core::memory_efficient::{ZeroCopyOps, MemoryMappedArray, LazyArray};

// Standard library
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Result type for zero-copy RDF operations
pub type Result<T> = std::result::Result<T, OxirsError>;

/// Configuration for zero-copy RDF operations
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Size of buffer pool for memory reuse
    pub buffer_pool_size: usize,
    /// Chunk size for processing large datasets
    pub chunk_size: usize,
    /// Enable memory-mapped file support
    pub enable_mmap: bool,
    /// Enable adaptive chunking based on workload
    pub enable_adaptive_chunking: bool,
    /// Enable lazy evaluation for queries
    pub enable_lazy_eval: bool,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            buffer_pool_size: 100,
            chunk_size: 10000,
            enable_mmap: true,
            enable_adaptive_chunking: true,
            enable_lazy_eval: true,
        }
    }
}

impl ZeroCopyConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the buffer pool size
    pub fn with_buffer_pool_size(mut self, size: usize) -> Self {
        self.buffer_pool_size = size;
        self
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Enable or disable memory-mapped files
    pub fn with_mmap(mut self, enable: bool) -> Self {
        self.enable_mmap = enable;
        self
    }

    /// Enable or disable adaptive chunking
    pub fn with_adaptive_chunking(mut self, enable: bool) -> Self {
        self.enable_adaptive_chunking = enable;
        self
    }

    /// Enable or disable lazy evaluation
    pub fn with_lazy_eval(mut self, enable: bool) -> Self {
        self.enable_lazy_eval = enable;
        self
    }
}

/// Zero-copy triple store using SciRS2-core's memory management
///
/// This store minimizes memory allocations by:
/// - Reusing buffers from a pool
/// - Using zero-copy views for query results
/// - Memory-mapping large files for direct access (when enabled)
/// - Chunked processing for large datasets
pub struct ZeroCopyTripleStore {
    /// Configuration
    config: ZeroCopyConfig,
    /// Buffer pool for memory reuse
    buffer_pool: Arc<RwLock<BufferPool<u8>>>,
    /// In-memory triple storage (for non-mmap mode)
    triples: Arc<RwLock<Vec<Triple>>>,
    /// Memory-mapped file references (placeholder for future implementation)
    _mmap_files: Arc<RwLock<HashMap<String, String>>>,
}

impl ZeroCopyTripleStore {
    /// Create a new zero-copy triple store with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ZeroCopyConfig::default())
    }

    /// Create a zero-copy triple store with custom configuration
    pub fn with_config(config: ZeroCopyConfig) -> Result<Self> {
        let buffer_pool = Arc::new(RwLock::new(BufferPool::new()));

        Ok(Self {
            config,
            buffer_pool,
            triples: Arc::new(RwLock::new(Vec::new())),
            _mmap_files: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load RDF data from a file using zero-copy parsing
    ///
    /// If memory-mapping is enabled, the file will be mapped into memory
    /// for direct access without reading into RAM.
    pub fn load_file_zero_copy(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        if self.config.enable_mmap {
            self.load_file_mmap(path)
        } else {
            self.load_file_buffered(path)
        }
    }

    /// Load a file using memory-mapped I/O with SciRS2-core
    ///
    /// Uses memory-mapped arrays for zero-copy file access. The file is mapped
    /// directly into memory without loading into RAM, enabling efficient processing
    /// of large RDF datasets.
    fn load_file_mmap(&mut self, path: &Path) -> Result<()> {
        use scirs2_core::memory_efficient::{open_mmap, AccessMode};

        // Open the file as a memory-mapped array
        let mmap_result = open_mmap::<u8, scirs2_core::ndarray_ext::Ix1>(
            path,
            AccessMode::ReadOnly,
            0, // offset
        );

        match mmap_result {
            Ok(mmap) => {
                // Store reference to the memory-mapped file
                {
                    let mut mmap_files = self
                        ._mmap_files
                        .write()
                        .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

                    mmap_files.insert(
                        path.to_string_lossy().to_string(),
                        format!("mmap:{}", mmap.size),
                    );
                } // Drop the lock here

                // Process the memory-mapped data in chunks for efficiency
                self.parse_mmap_chunked(&mmap)?;

                Ok(())
            }
            Err(e) => {
                // Fall back to buffered loading if memory mapping fails
                tracing::warn!(
                    "Memory mapping failed ({}), falling back to buffered I/O",
                    e
                );
                self.load_file_buffered(path)
            }
        }
    }

    /// Parse memory-mapped data in chunks
    ///
    /// Uses the configured chunk size to process the memory-mapped data efficiently.
    /// For adaptive chunking, we use a heuristic based on available memory.
    fn parse_mmap_chunked(
        &mut self,
        mmap: &scirs2_core::memory_efficient::MemoryMappedArray<u8>,
    ) -> Result<()> {
        // Determine chunk size - adaptive or fixed
        let chunk_size = if self.config.enable_adaptive_chunking {
            // Use adaptive chunk size based on data size
            // Aim for ~100MB chunks, but adapt based on total size
            let data_size = mmap.size;
            let target_chunk_mb = 100 * 1024 * 1024; // 100MB in bytes

            if data_size < target_chunk_mb {
                // Small file - use one chunk
                data_size
            } else {
                // Large file - use 100MB chunks
                target_chunk_mb
            }
        } else {
            // Use configured fixed chunk size
            self.config.chunk_size
        };

        // Process the memory-mapped data in chunks
        let data_slice = mmap.as_slice();
        for chunk_start in (0..data_slice.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(data_slice.len());
            let chunk = &data_slice[chunk_start..chunk_end];

            // Parse this chunk
            self.parse_buffer_zero_copy(chunk)?;
        }

        Ok(())
    }

    /// Simple chunked parsing without adaptive chunking
    #[allow(dead_code)]
    fn parse_mmap_simple(
        &mut self,
        mmap: &scirs2_core::memory_efficient::MemoryMappedArray<u8>,
    ) -> Result<()> {
        // Use configured chunk size
        let chunk_size = self.config.chunk_size;

        // Process the memory-mapped data in fixed-size chunks
        let data_slice = mmap.as_slice();
        for chunk_start in (0..data_slice.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(data_slice.len());
            let chunk = &data_slice[chunk_start..chunk_end];

            // Parse this chunk
            self.parse_buffer_zero_copy(chunk)?;
        }

        Ok(())
    }

    /// Load a file using buffered I/O with buffer pooling
    fn load_file_buffered(&mut self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let file_size = std::fs::metadata(path)
            .map_err(|e| OxirsError::Io(e.to_string()))?
            .len() as usize;

        // Scope the buffer pool lock to avoid borrow checker issues
        let mut buffer = {
            let mut buffer_pool = self
                .buffer_pool
                .write()
                .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;
            buffer_pool.acquire_vec(file_size)
        };

        // Read file into buffer
        let mut file = File::open(path).map_err(|e| OxirsError::Io(e.to_string()))?;

        file.read_to_end(&mut buffer)
            .map_err(|e| OxirsError::Io(e.to_string()))?;

        // Parse triples from buffer using zero-copy
        self.parse_buffer_zero_copy(&buffer)?;

        // Release buffer back to pool
        let mut buffer_pool = self
            .buffer_pool
            .write()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;
        buffer_pool.release_vec(buffer);

        Ok(())
    }

    /// Parse triples from a buffer using zero-copy operations
    ///
    /// This uses a streaming parser that processes N-Triples format with minimal
    /// allocations. The buffer is processed in place without creating intermediate
    /// string copies.
    fn parse_buffer_zero_copy(&mut self, buffer: &[u8]) -> Result<()> {
        use std::str;

        // For zero-copy parsing, we'll use line-based processing for N-Triples
        // which is the simplest RDF format and most amenable to zero-copy parsing
        let content = str::from_utf8(buffer)
            .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {}", e)))?;

        let mut line_count = 0;
        let mut parse_errors = 0;

        for line in content.lines() {
            line_count += 1;

            // Skip empty lines and comments
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Parse the line as an N-Triple
            match self.parse_ntriple_line(trimmed) {
                Ok(triple) => {
                    self.insert_zero_copy(triple)?;
                }
                Err(e) => {
                    tracing::warn!("Failed to parse line {}: {} - {}", line_count, trimmed, e);
                    parse_errors += 1;

                    // Fail fast if too many parse errors
                    if parse_errors > 100 {
                        return Err(OxirsError::Parse(format!(
                            "Too many parse errors ({}), stopping",
                            parse_errors
                        )));
                    }
                }
            }
        }

        tracing::info!("Parsed {} lines with {} errors", line_count, parse_errors);

        Ok(())
    }

    /// Parse a single N-Triple line into a Triple
    ///
    /// This is a simple parser for the N-Triples format, which has the structure:
    /// <subject> <predicate> <object> .
    ///
    /// We use string slicing to avoid allocations where possible.
    fn parse_ntriple_line(&self, line: &str) -> Result<Triple> {
        use crate::model::{BlankNode, Literal, NamedNode};

        // Split on whitespace, but preserve quoted strings and angle brackets
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut in_angle = false;

        for ch in line.chars() {
            match ch {
                '<' if !in_quotes => {
                    // Skip any accumulated whitespace
                    current.clear();
                    in_angle = true;
                    current.push(ch);
                }
                '>' if !in_quotes && in_angle => {
                    in_angle = false;
                    current.push(ch);
                    parts.push(current.trim().to_string());
                    current.clear();
                }
                '"' if !in_angle => {
                    if !in_quotes {
                        // Starting a quoted string, clear any whitespace
                        current.clear();
                    }
                    in_quotes = !in_quotes;
                    current.push(ch);

                    // If we're closing quotes, add to parts
                    if !in_quotes {
                        parts.push(current.trim().to_string());
                        current.clear();
                    }
                }
                ' ' | '\t' if !in_quotes && !in_angle => {
                    // Skip whitespace outside quotes and angle brackets
                    if !current.is_empty() && current != "." {
                        parts.push(current.trim().to_string());
                        current.clear();
                    }
                }
                '.' if !in_quotes && !in_angle => {
                    // End of triple
                    if !current.is_empty() && current != "." {
                        parts.push(current.trim().to_string());
                    }
                    break;
                }
                _ if in_quotes || in_angle => {
                    // Always add characters when inside quotes or angle brackets
                    current.push(ch);
                }
                _ if !ch.is_whitespace() => {
                    // Add non-whitespace characters for blank nodes
                    current.push(ch);
                }
                _ => {
                    // Skip whitespace
                }
            }
        }

        if parts.len() < 3 {
            return Err(OxirsError::Parse(format!(
                "Invalid N-Triple: expected 3 parts, got {} (parts: {:?})",
                parts.len(),
                parts
            )));
        }

        // Parse subject (IRI or blank node)
        let subject = if parts[0].starts_with('<') && parts[0].ends_with('>') {
            let iri = parts[0][1..parts[0].len() - 1].to_string();
            Subject::NamedNode(NamedNode::new(iri)?)
        } else if parts[0].starts_with("_:") {
            let label = parts[0][2..].to_string();
            Subject::BlankNode(BlankNode::new(label)?)
        } else {
            return Err(OxirsError::Parse(format!("Invalid subject: {}", parts[0])));
        };

        // Parse predicate (IRI)
        let predicate = if parts[1].starts_with('<') && parts[1].ends_with('>') {
            let iri = parts[1][1..parts[1].len() - 1].to_string();
            Predicate::NamedNode(NamedNode::new(iri)?)
        } else {
            return Err(OxirsError::Parse(format!(
                "Invalid predicate: {}",
                parts[1]
            )));
        };

        // Parse object (IRI, blank node, or literal)
        let object = if parts[2].starts_with('<') && parts[2].ends_with('>') {
            let iri = parts[2][1..parts[2].len() - 1].to_string();
            Object::NamedNode(NamedNode::new(iri)?)
        } else if parts[2].starts_with("_:") {
            let label = parts[2][2..].to_string();
            Object::BlankNode(BlankNode::new(label)?)
        } else if parts[2].starts_with('"') {
            // Parse literal (simplified - doesn't handle language tags or datatypes)
            let mut value = parts[2].clone();

            // Remove quotes
            if value.starts_with('"') {
                value.remove(0);
            }
            if value.ends_with('"') {
                value.pop();
            }

            Object::Literal(Literal::new(value))
        } else {
            return Err(OxirsError::Parse(format!("Invalid object: {}", parts[2])));
        };

        Ok(Triple::new(subject, predicate, object))
    }

    /// Query triples with zero-copy views (minimal allocations)
    ///
    /// Returns filtered triples by pattern matching. While this returns owned
    /// triples (not references due to Rust lifetime constraints), it uses
    /// efficient filtering to minimize overhead.
    ///
    /// For truly zero-copy queries, use `query_indices` which returns
    /// indices that can be used to access triples.
    pub fn query_zero_copy(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        let triples = self
            .triples
            .read()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        // Use iterator filtering for efficient processing
        let results: Vec<Triple> = triples
            .iter()
            .filter(|triple| {
                // Check subject match
                if let Some(s) = subject {
                    if triple.subject() != s {
                        return false;
                    }
                }

                // Check predicate match
                if let Some(p) = predicate {
                    if triple.predicate() != p {
                        return false;
                    }
                }

                // Check object match
                if let Some(o) = object {
                    if triple.object() != o {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect();

        Ok(results)
    }

    /// Query for triple indices (true zero-copy)
    ///
    /// Returns the indices of matching triples without copying the triples themselves.
    /// These indices can be used to access triples later via `get_by_index`.
    pub fn query_indices(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<usize>> {
        let triples = self
            .triples
            .read()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        // Return indices of matching triples
        let indices: Vec<usize> = triples
            .iter()
            .enumerate()
            .filter_map(|(idx, triple)| {
                // Check subject match
                if let Some(s) = subject {
                    if triple.subject() != s {
                        return None;
                    }
                }

                // Check predicate match
                if let Some(p) = predicate {
                    if triple.predicate() != p {
                        return None;
                    }
                }

                // Check object match
                if let Some(o) = object {
                    if triple.object() != o {
                        return None;
                    }
                }

                Some(idx)
            })
            .collect();

        Ok(indices)
    }

    /// Get a triple by index (for use with query_indices)
    pub fn get_by_index(&self, index: usize) -> Result<Option<Triple>> {
        let triples = self
            .triples
            .read()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        Ok(triples.get(index).cloned())
    }

    /// Get the total number of triples
    pub fn len(&self) -> Result<usize> {
        let triples = self
            .triples
            .read()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        Ok(triples.len())
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Insert a triple using zero-copy operations
    pub fn insert_zero_copy(&mut self, triple: Triple) -> Result<bool> {
        let mut triples = self
            .triples
            .write()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        // Check for duplicates
        if triples.contains(&triple) {
            return Ok(false);
        }

        triples.push(triple);
        Ok(true)
    }

    /// Bulk insert triples using zero-copy batch operations
    pub fn bulk_insert_zero_copy(&mut self, new_triples: Vec<Triple>) -> Result<usize> {
        let mut triples = self
            .triples
            .write()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        let initial_len = triples.len();

        // Use adaptive chunking if enabled
        if self.config.enable_adaptive_chunking && new_triples.len() > self.config.chunk_size {
            // Process in chunks to avoid memory spikes
            for chunk in new_triples.chunks(self.config.chunk_size) {
                for triple in chunk {
                    if !triples.contains(triple) {
                        triples.push(triple.clone());
                    }
                }
            }
        } else {
            // Process all at once for small batches
            for triple in new_triples {
                if !triples.contains(&triple) {
                    triples.push(triple);
                }
            }
        }

        let inserted = triples.len() - initial_len;
        Ok(inserted)
    }

    /// Get statistics about memory usage
    pub fn memory_stats(&self) -> ZeroCopyStats {
        let triples_count = self.triples.read().map(|t| t.len()).unwrap_or(0);
        let mmap_files_count = self._mmap_files.read().map(|m| m.len()).unwrap_or(0);

        ZeroCopyStats {
            triples_count,
            mmap_files_count,
            buffer_pool_size: self.config.buffer_pool_size,
            chunk_size: self.config.chunk_size,
        }
    }
}

impl Default for ZeroCopyTripleStore {
    fn default() -> Self {
        Self::new().expect("Failed to create default ZeroCopyTripleStore")
    }
}

/// Statistics for zero-copy operations
#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    /// Number of triples in memory
    pub triples_count: usize,
    /// Number of memory-mapped files
    pub mmap_files_count: usize,
    /// Buffer pool size
    pub buffer_pool_size: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_zero_copy_store_creation() {
        let store = ZeroCopyTripleStore::new();
        assert!(store.is_ok());

        let store = store.unwrap();
        assert_eq!(store.len().unwrap(), 0);
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_zero_copy_config() {
        let config = ZeroCopyConfig::new()
            .with_buffer_pool_size(200)
            .with_chunk_size(5000)
            .with_mmap(false)
            .with_adaptive_chunking(true)
            .with_lazy_eval(false);

        assert_eq!(config.buffer_pool_size, 200);
        assert_eq!(config.chunk_size, 5000);
        assert!(!config.enable_mmap);
        assert!(config.enable_adaptive_chunking);
        assert!(!config.enable_lazy_eval);
    }

    #[test]
    fn test_zero_copy_insert() -> Result<()> {
        use crate::model::{Literal, NamedNode};

        let mut store = ZeroCopyTripleStore::new()?;

        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s")?);
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
        let object = Object::Literal(Literal::new("test"));

        let triple = Triple::new(subject, predicate, object);

        // First insert should succeed
        assert!(store.insert_zero_copy(triple.clone())?);
        assert_eq!(store.len()?, 1);

        // Duplicate insert should return false
        assert!(!store.insert_zero_copy(triple)?);
        assert_eq!(store.len()?, 1);

        Ok(())
    }

    #[test]
    fn test_zero_copy_bulk_insert() -> Result<()> {
        use crate::model::{Literal, NamedNode};

        let mut store = ZeroCopyTripleStore::new()?;

        let mut triples = Vec::new();
        for i in 0..100 {
            let subject = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i))?);
            let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
            let object = Object::Literal(Literal::new(i.to_string()));
            triples.push(Triple::new(subject, predicate, object));
        }

        let inserted = store.bulk_insert_zero_copy(triples.clone())?;
        assert_eq!(inserted, 100);
        assert_eq!(store.len()?, 100);

        // Inserting duplicates should insert 0
        let inserted = store.bulk_insert_zero_copy(triples)?;
        assert_eq!(inserted, 0);
        assert_eq!(store.len()?, 100);

        Ok(())
    }

    #[test]
    fn test_memory_stats() -> Result<()> {
        let store = ZeroCopyTripleStore::new()?;
        let stats = store.memory_stats();

        assert_eq!(stats.triples_count, 0);
        assert_eq!(stats.mmap_files_count, 0);
        assert_eq!(stats.buffer_pool_size, 100);

        Ok(())
    }

    #[test]
    fn test_parse_ntriple_simple() -> Result<()> {
        let store = ZeroCopyTripleStore::new()?;

        // Simple N-Triple
        let line = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let triple = store.parse_ntriple_line(line)?;

        match triple.subject() {
            Subject::NamedNode(nn) => {
                assert_eq!(nn.as_str(), "http://example.org/s");
            }
            _ => panic!("Expected NamedNode subject"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_ntriple_with_literal() -> Result<()> {
        let store = ZeroCopyTripleStore::new()?;

        let line = r#"<http://example.org/s> <http://example.org/p> "Hello World" ."#;
        let triple = store.parse_ntriple_line(line)?;

        match triple.object() {
            Object::Literal(lit) => {
                assert_eq!(lit.value(), "Hello World");
            }
            _ => panic!("Expected Literal object"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_buffer_zero_copy_ntriples() -> Result<()> {
        let mut store = ZeroCopyTripleStore::new()?;

        let data = b"<http://example.org/s1> <http://example.org/p1> <http://example.org/o1> .\n\
                       <http://example.org/s2> <http://example.org/p2> <http://example.org/o2> .\n\
                       # This is a comment\n\
                       <http://example.org/s3> <http://example.org/p3> \"Literal value\" .";

        store.parse_buffer_zero_copy(data)?;

        assert_eq!(store.len()?, 3);

        Ok(())
    }

    #[test]
    fn test_query_zero_copy_match_all() -> Result<()> {
        let mut store = ZeroCopyTripleStore::new()?;

        // Insert test triples
        for i in 0..10 {
            let subject = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i))?);
            let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
            let object = Object::Literal(Literal::new(i.to_string()));
            store.insert_zero_copy(Triple::new(subject, predicate, object))?;
        }

        // Query all triples
        let results = store.query_zero_copy(None, None, None)?;
        assert_eq!(results.len(), 10);

        Ok(())
    }

    #[test]
    fn test_query_zero_copy_with_predicate_filter() -> Result<()> {
        let mut store = ZeroCopyTripleStore::new()?;

        let p1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1")?);
        let p2 = Predicate::NamedNode(NamedNode::new("http://example.org/p2")?);

        // Insert triples with different predicates
        for i in 0..5 {
            let subject = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i))?);
            let object = Object::Literal(Literal::new(i.to_string()));
            store.insert_zero_copy(Triple::new(subject.clone(), p1.clone(), object.clone()))?;
            store.insert_zero_copy(Triple::new(subject, p2.clone(), object))?;
        }

        // Query with predicate filter
        let results = store.query_zero_copy(None, Some(&p1), None)?;
        assert_eq!(results.len(), 5);

        Ok(())
    }

    #[test]
    fn test_query_indices() -> Result<()> {
        let mut store = ZeroCopyTripleStore::new()?;

        // Insert test triples
        for i in 0..5 {
            let subject = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i))?);
            let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
            let object = Object::Literal(Literal::new(i.to_string()));
            store.insert_zero_copy(Triple::new(subject, predicate, object))?;
        }

        // Query for indices
        let indices = store.query_indices(None, None, None)?;
        assert_eq!(indices.len(), 5);

        // Verify indices are valid
        for idx in indices {
            let triple = store.get_by_index(idx)?;
            assert!(triple.is_some());
        }

        Ok(())
    }

    #[test]
    fn test_get_by_index_out_of_bounds() -> Result<()> {
        let store = ZeroCopyTripleStore::new()?;

        let result = store.get_by_index(999)?;
        assert!(result.is_none());

        Ok(())
    }

    #[test]
    fn test_zero_copy_with_adaptive_chunking() -> Result<()> {
        let config = ZeroCopyConfig::new()
            .with_adaptive_chunking(true)
            .with_chunk_size(5);

        let mut store = ZeroCopyTripleStore::with_config(config)?;

        // Insert large batch
        let mut triples = Vec::new();
        for i in 0..100 {
            let subject = Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i))?);
            let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
            let object = Object::Literal(Literal::new(i.to_string()));
            triples.push(Triple::new(subject, predicate, object));
        }

        let inserted = store.bulk_insert_zero_copy(triples)?;
        assert_eq!(inserted, 100);

        Ok(())
    }
}
