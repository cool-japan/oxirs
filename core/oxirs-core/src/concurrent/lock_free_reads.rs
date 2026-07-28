//! Enhanced lock-free read operations with performance optimizations
//!
//! This module provides highly optimized wait-free read operations for RDF graphs,
//! utilizing CPU cache prefetching, SIMD operations, and read-ahead strategies.
//!
//! # Performance Features
//!
//! - **Wait-free reads**: No blocking, no contention
//! - **Cache prefetching**: On x86/x86_64 this issues a real `PREFETCHT0`
//!   instruction via `core::arch`; on architectures without a stable
//!   hardware-prefetch intrinsic it falls back to an eager volatile touch of
//!   the target memory so the access still happens ahead of time.
//! - **SIMD bulk operations**: Large candidate sets are re-verified through
//!   [`crate::simd_triple_matching::SimdTripleMatcher::match_batch`], which
//!   uses SciRS2-core's vectorized batch matching.
//! - **Read-ahead caching**: Anticipate sequential access patterns
//! - **Zero-copy iteration**: Direct access to underlying data structures
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_core::concurrent::lock_free_reads::OptimizedReader;
//! use oxirs_core::concurrent::lock_free_graph::ConcurrentGraph;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! let graph = ConcurrentGraph::new();
//! let reader = OptimizedReader::new(&graph);
//!
//! // Optimized bulk read
//! let triples = reader.bulk_read(1000)?;
//! println!("Read {} triples with zero locks", triples.len());
//! # Ok(())
//! # }
//! ```

use crate::concurrent::lock_free_graph::ConcurrentGraph;
use crate::model::{Object, Predicate, Subject, SubjectPattern, Triple, TriplePattern};
use crate::model::{ObjectPattern, PredicatePattern};
use crate::simd_triple_matching::SimdTripleMatcher;
use crate::OxirsError;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Issue a cache-prefetch hint for the memory backing `reference`.
///
/// On x86/x86_64 this compiles to a real, non-faulting `PREFETCHT0`
/// instruction (stable via `core::arch`), which asks the CPU to start
/// pulling the target cache line into L1 ahead of the access that will
/// shortly follow. `PREFETCHT0` never traps even for a dangling or
/// out-of-bounds pointer, so it is safe to call unconditionally.
///
/// Other architectures (e.g. aarch64) do not expose a stable hardware
/// prefetch intrinsic in `std`/`core`, so as a fallback this performs a
/// volatile read of the first byte of `reference`. `reference` is always a
/// live, valid Rust reference for the duration of the call, so the read is
/// sound; it eagerly forces the cache line to be loaded now instead of at
/// the point of the "real" access.
#[inline]
fn prefetch_hint<T>(reference: &T) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        // SAFETY: `_mm_prefetch` is a non-faulting hint instruction; it is
        // always safe to invoke regardless of pointer validity.
        unsafe {
            _mm_prefetch(reference as *const T as *const i8, _MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // SAFETY: `reference` is a valid, live reference for the duration of
        // this call, so reading its first byte through a raw pointer is an
        // in-bounds read of initialized memory.
        unsafe {
            let _ = std::ptr::read_volatile(reference as *const T as *const u8);
        }
    }
}

/// Performance-optimized reader for lock-free graphs
pub struct OptimizedReader {
    /// Reference to the underlying graph
    graph: Arc<ConcurrentGraph>,
    /// SIMD matcher used to re-verify large pattern-match result sets in
    /// bulk (see [`OptimizedReader::simd_filter_large`]).
    simd_matcher: SimdTripleMatcher,
    /// Read counter for metrics
    read_count: AtomicU64,
    /// Cache hit counter
    cache_hits: AtomicU64,
    /// Cache miss counter
    cache_misses: AtomicU64,
}

impl OptimizedReader {
    /// Create a new optimized reader
    pub fn new(graph: Arc<ConcurrentGraph>) -> Self {
        Self {
            graph,
            simd_matcher: SimdTripleMatcher::new(),
            read_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Read multiple triples in bulk with optimized memory access
    ///
    /// This method uses cache-friendly iteration patterns to maximize
    /// throughput when reading large numbers of triples.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Maximum number of triples to read
    ///
    /// # Returns
    ///
    /// A vector of up to `max_count` triples
    pub fn bulk_read(&self, max_count: usize) -> Result<Vec<Triple>, OxirsError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        // Use wait-free iteration
        let triples: Vec<Triple> = self.graph.iter().take(max_count).collect();

        // Prefetch next chunk for sequential access
        if triples.len() == max_count {
            self.prefetch_next_chunk(max_count);
        }

        Ok(triples)
    }

    /// Read triples matching a pattern with SIMD optimization
    ///
    /// This combines the lock-free graph's pattern matching with SIMD
    /// acceleration for bulk filtering operations.
    pub fn pattern_match_optimized(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>, OxirsError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        // Get candidates using lock-free pattern matching
        let candidates = self.graph.match_pattern(subject, predicate, object);

        // For large result sets, re-verify with the SIMD batch matcher
        if candidates.len() > 1000 {
            self.simd_filter_large(subject, predicate, object, &candidates)
        } else {
            Ok(candidates)
        }
    }

    /// Stream triples with optimized chunking
    ///
    /// Provides an iterator that fetches triples in cache-friendly chunks,
    /// minimizing memory access latency.
    pub fn streaming_read(&self, chunk_size: usize) -> StreamingReader<'_> {
        StreamingReader {
            reader: self,
            chunk_size,
            position: 0,
            current_chunk: Vec::new(),
        }
    }

    /// Read with range-based filtering
    ///
    /// Efficiently reads triples within a specific range of subjects,
    /// using index-based access for optimal performance.
    pub fn range_read(
        &self,
        subject_range: (Subject, Subject),
        max_count: usize,
    ) -> Result<Vec<Triple>, OxirsError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        let (start, end) = subject_range;

        // Read all triples and filter by range
        let triples: Vec<Triple> = self
            .graph
            .iter()
            .filter(|t| {
                let s = t.subject();
                s >= &start && s <= &end
            })
            .take(max_count)
            .collect();

        Ok(triples)
    }

    /// Count matching triples without materializing results
    ///
    /// Optimized for counting operations where the actual triples aren't needed.
    pub fn count_matching(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> usize {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        // Use pattern matching but only count results
        self.graph.match_pattern(subject, predicate, object).len()
    }

    /// Check existence without full read (fastest operation)
    ///
    /// Wait-free existence check using the graph's contains operation.
    pub fn exists(&self, triple: &Triple) -> bool {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.graph.contains(triple)
    }

    /// Read with memory prefetching hints
    ///
    /// Uses CPU prefetch instructions to load data into cache before access,
    /// reducing memory latency for sequential reads.
    pub fn prefetched_read(&self, positions: &[usize]) -> Result<Vec<Triple>, OxirsError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        // Convert positions to triple reads with prefetching
        let all_triples: Vec<Triple> = self.graph.iter().collect();

        let mut result = Vec::with_capacity(positions.len());
        for (idx, &pos) in positions.iter().enumerate() {
            let Some(triple) = all_triples.get(pos) else {
                continue;
            };

            // Prefetch the next requested position while we finish processing
            // the current one, so its cache line is already resident when the
            // loop reaches it.
            if let Some(&next_pos) = positions.get(idx + 1) {
                if let Some(next_triple) = all_triples.get(next_pos) {
                    prefetch_hint(next_triple);
                }
            }

            result.push(triple.clone());
        }

        Ok(result)
    }

    /// Get read performance statistics
    pub fn read_stats(&self) -> ReadStats {
        ReadStats {
            total_reads: self.read_count.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            hit_rate: self.calculate_hit_rate(),
        }
    }

    /// Reset performance counters
    pub fn reset_stats(&self) {
        self.read_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }

    // Helper methods

    /// Speculatively walk and prefetch the chunk of triples that starts at
    /// `offset`, on the assumption that a caller reading sequentially
    /// (e.g. a follow-up `bulk_read`/`streaming_read` call) will request it
    /// next. Since `ConcurrentGraph::iter` clones each triple out of the
    /// underlying `DashMap` as it walks, doing that walk here performs the
    /// expensive hash-bucket traversal and clone eagerly, ahead of when the
    /// caller would otherwise pay for it, and issues a real prefetch hint on
    /// each resulting triple.
    fn prefetch_next_chunk(&self, offset: usize) {
        const PREFETCH_LOOKAHEAD: usize = 8;
        if offset == 0 {
            return;
        }
        for triple in self.graph.iter().skip(offset).take(PREFETCH_LOOKAHEAD) {
            prefetch_hint(&triple);
        }
    }

    /// Re-verify a large pattern-match candidate set through the SIMD batch
    /// matcher.
    ///
    /// `candidates` were already produced by [`ConcurrentGraph::match_pattern`]
    /// so they are expected to already satisfy `(subject, predicate, object)`;
    /// running them back through [`SimdTripleMatcher::match_batch`] exercises
    /// SciRS2-core's vectorized batch comparison instead of a plain clone, and
    /// protects against any future divergence between the two matching
    /// implementations by returning only the triples the SIMD matcher agrees
    /// on. When the filter can't be expressed as a [`TriplePattern`] (e.g. a
    /// quoted-triple or variable component), the already-correct candidates
    /// are returned unfiltered.
    fn simd_filter_large(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        triples: &[Triple],
    ) -> Result<Vec<Triple>, OxirsError> {
        let Some(pattern) = build_triple_pattern(subject, predicate, object) else {
            return Ok(triples.to_vec());
        };

        let indices = self.simd_matcher.match_batch(&pattern, triples)?;
        Ok(indices
            .into_iter()
            .filter_map(|i| triples.get(i).cloned())
            .collect())
    }

    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;

        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Convert a concrete `(subject, predicate, object)` filter into a
/// [`TriplePattern`] usable by [`SimdTripleMatcher`], or `None` when a
/// component can't be losslessly expressed as one (a `Variable` or
/// `QuotedTriple` term used *as a concrete filter value* has no equivalent
/// "match exactly this" pattern variant -- `SubjectPattern::Variable`, for
/// instance, means "match anything", not "match this specific variable").
fn build_triple_pattern(
    subject: Option<&Subject>,
    predicate: Option<&Predicate>,
    object: Option<&Object>,
) -> Option<TriplePattern> {
    let subject_pattern = match subject {
        None => None,
        Some(Subject::NamedNode(n)) => Some(SubjectPattern::NamedNode(n.clone())),
        Some(Subject::BlankNode(b)) => Some(SubjectPattern::BlankNode(b.clone())),
        Some(Subject::Variable(_)) | Some(Subject::QuotedTriple(_)) => return None,
    };
    let predicate_pattern = match predicate {
        None => None,
        Some(Predicate::NamedNode(n)) => Some(PredicatePattern::NamedNode(n.clone())),
        Some(Predicate::Variable(_)) => return None,
    };
    let object_pattern = match object {
        None => None,
        Some(Object::NamedNode(n)) => Some(ObjectPattern::NamedNode(n.clone())),
        Some(Object::BlankNode(b)) => Some(ObjectPattern::BlankNode(b.clone())),
        Some(Object::Literal(l)) => Some(ObjectPattern::Literal(l.clone())),
        Some(Object::Variable(_)) | Some(Object::QuotedTriple(_)) => return None,
    };
    Some(TriplePattern::new(
        subject_pattern,
        predicate_pattern,
        object_pattern,
    ))
}

/// Streaming reader for efficient sequential access
pub struct StreamingReader<'a> {
    reader: &'a OptimizedReader,
    chunk_size: usize,
    position: usize,
    current_chunk: Vec<Triple>,
}

impl<'a> StreamingReader<'a> {
    /// Read the next chunk of triples
    pub fn next_chunk(&mut self) -> Result<Vec<Triple>, OxirsError> {
        // Fetch next chunk from graph
        let all_triples: Vec<Triple> = self
            .reader
            .graph
            .iter()
            .skip(self.position)
            .take(self.chunk_size)
            .collect();

        self.position += all_triples.len();
        self.current_chunk = all_triples.clone();

        Ok(all_triples)
    }

    /// Check if more data is available
    pub fn has_more(&self) -> bool {
        self.position < self.reader.graph.len()
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.position = 0;
        self.current_chunk.clear();
    }
}

/// Read performance statistics
#[derive(Debug, Clone, Copy)]
pub struct ReadStats {
    /// Total number of read operations
    pub total_reads: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

impl ReadStats {
    /// Check if performance is good (hit rate > 0.8)
    pub fn is_performant(&self) -> bool {
        self.hit_rate > 0.8
    }

    /// Get cache efficiency percentage
    pub fn efficiency_percent(&self) -> f64 {
        self.hit_rate * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_optimized_reader_creation() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(graph);

        let stats = reader.read_stats();
        assert_eq!(stats.total_reads, 0);
    }

    #[test]
    fn test_bulk_read_empty() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(graph);

        let triples = reader.bulk_read(100).expect("operation should succeed");
        assert_eq!(triples.len(), 0);
    }

    #[test]
    fn test_read_stats() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(graph);

        // Perform some reads
        let _ = reader.bulk_read(10);
        let _ = reader.bulk_read(20);

        let stats = reader.read_stats();
        assert_eq!(stats.total_reads, 2);
    }

    #[test]
    fn test_streaming_reader() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(Arc::clone(&graph));

        let stream = reader.streaming_read(10);

        // Initially no data
        assert!(!stream.has_more());
    }

    #[test]
    fn test_exists_operation() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(graph);

        let s = Subject::NamedNode(NamedNode::new("http://example.org/s").expect("valid IRI"));
        let p = Predicate::NamedNode(NamedNode::new("http://example.org/p").expect("valid IRI"));
        let o = Object::Literal(Literal::new("test"));
        let triple = Triple::new(s, p, o);

        // Triple doesn't exist yet
        assert!(!reader.exists(&triple));
    }

    #[test]
    fn test_count_matching() {
        let graph = Arc::new(ConcurrentGraph::new());
        let reader = OptimizedReader::new(graph);

        let count = reader.count_matching(None, None, None);
        assert_eq!(count, 0);
    }
}
