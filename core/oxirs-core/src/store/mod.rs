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

pub use indexed_graph::{IndexedGraph, IndexStats, MemoryUsage, IndexType};
pub use term_interner::{TermInterner, InternerStats};
pub use mmap_store::{MmapStore, StoreStats};
pub use mmap_index::{MmapIndex, IndexEntry};
pub use arena::{
    LocalArena, ConcurrentArena, GraphArena, ScopedArena,
    ArenaStr, ArenaTerm, ArenaTriple,
};

/// Re-export commonly used types
pub use indexed_graph::InternedTriple;