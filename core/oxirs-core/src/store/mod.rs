//! High-performance store module with multi-index graph and term interning
//!
//! This module provides optimized storage structures for RDF data:
//! - Multi-indexed graph with SPO/POS/OSP indexes for O(log n) lookups
//! - Term interning for memory-efficient string storage
//! - Support for concurrent read access
//! - Batch operations for improved performance
//! - Memory-mapped storage for datasets larger than RAM

pub mod indexed_graph;
pub mod term_interner;
pub mod mmap_store;
pub mod mmap_index;
pub mod arena;
pub mod encoding;
pub mod binary;
pub mod adaptive_index;

pub use indexed_graph::{IndexedGraph, IndexStats, MemoryUsage, IndexType};
pub use term_interner::{TermInterner, InternerStats};
pub use mmap_store::{MmapStore, StoreStats};
pub use mmap_index::{MmapIndex, IndexEntry};
pub use arena::{
    LocalArena, ConcurrentArena, GraphArena, ScopedArena,
    ArenaStr, ArenaTerm, ArenaTriple,
};
pub use encoding::{
    EncodedTerm, EncodedTriple, EncodedQuad, StrHash, SmallString,
};
pub use binary::{
    encode_term, decode_term, QuadEncoding, WRITTEN_TERM_MAX_SIZE,
};
pub use adaptive_index::{
    AdaptiveIndexManager, AdaptiveConfig, QueryPattern, AdaptiveIndexStats,
};

/// Re-export commonly used types
pub use indexed_graph::InternedTriple;