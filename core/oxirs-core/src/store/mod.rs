//! High-performance store module with multi-index graph and term interning
//!
//! This module provides optimized storage structures for RDF data:
//! - Multi-indexed graph with SPO/POS/OSP indexes for O(log n) lookups
//! - Term interning for memory-efficient string storage
//! - Support for concurrent read access
//! - Batch operations for improved performance
//! - Memory-mapped storage for datasets larger than RAM

pub mod adaptive_index;
pub mod arena;
pub mod binary;
pub mod encoding;
pub mod indexed_graph;
pub mod mmap_index;
pub mod mmap_store;
pub mod term_interner;

pub use adaptive_index::{AdaptiveConfig, AdaptiveIndexManager, AdaptiveIndexStats, QueryPattern};
pub use arena::{
    ArenaStr, ArenaTerm, ArenaTriple, ConcurrentArena, GraphArena, LocalArena, ScopedArena,
};
pub use binary::{decode_term, encode_term, QuadEncoding, WRITTEN_TERM_MAX_SIZE};
pub use encoding::{EncodedQuad, EncodedTerm, EncodedTriple, SmallString, StrHash};
pub use indexed_graph::{IndexStats, IndexType, IndexedGraph, MemoryUsage};
pub use mmap_index::{IndexEntry, MmapIndex};
pub use mmap_store::{MmapStore, StoreStats};
pub use term_interner::{InternerStats, TermInterner};

/// Re-export commonly used types
pub use indexed_graph::InternedTriple;
