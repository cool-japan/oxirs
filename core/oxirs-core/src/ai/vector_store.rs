//! Vector Store for Efficient Similarity Search — thin facade module.
//!
//! This module provides high-performance vector storage and similarity search
//! capabilities optimized for knowledge graph embeddings and AI applications.
//!
//! The implementation is split across sibling modules to keep each file within
//! the workspace size policy:
//! - [`crate::ai::vector_store_types`]  — traits, data structures, config and
//!   statistics types (`VectorStore`, `VectorData`, `VectorQuery`, `Filter`,
//!   `SimilarityMetric`, `VectorStoreConfig`, `IndexType`, `VectorIndex`, …).
//! - [`crate::ai::vector_store_index`]  — the built-in `FlatIndex` and
//!   `HNSWIndex` ANN backends.
//! - [`crate::ai::vector_store_search`] — SIMD-accelerated similarity kernels,
//!   the `InMemoryVectorStore`, performance metrics and the
//!   `create_vector_store` factory.
//!
//! The IVF, LSH and PQ index backends live in the dedicated sibling modules
//! (`ivf_index`, `lsh_index`, `pq_index`) and are re-exported here for
//! ergonomic access by callers.

pub use super::vector_store_index::*;
pub use super::vector_store_search::*;
pub use super::vector_store_types::*;

pub use super::ivf_index::IVFIndex;
pub use super::lsh_index::LSHIndex;
pub use super::pq_index::PQIndexLocal;
