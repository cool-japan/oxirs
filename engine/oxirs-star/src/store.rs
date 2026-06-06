//! RDF-star storage implementation with efficient handling of quoted triples.
//!
//! This module provides storage backends for RDF-star data, extending the core
//! OxiRS storage with support for quoted triples and efficient indexing.
//!
//! Features:
//! - B-tree indexing for efficient quoted triple lookups
//! - Bulk insertion optimizations for large datasets
//! - Memory-mapped storage options for persistent storage
//! - Compression for quoted triple storage
//! - Connection pooling for concurrent access
//! - Cache optimization strategies
//! - Transaction support with ACID properties
//!
//! The implementation is split across sibling modules for maintainability:
//! - [`crate::store_core`]: [`StarStore`] type, basic CRUD, and config helpers.
//! - [`crate::store_query`]: pattern queries, iteration, and core-RDF conversions.
//! - [`crate::store_indexing`]: index maintenance, bulk insert, statistics, pooling.
//!
//! This file hosts the shared internal submodule wiring and re-exports the
//! public surface of those siblings.

// Internal store submodules (visible to sibling files)
#[path = "store/bulk_insert.rs"]
pub(crate) mod bulk_insert_mod;
#[path = "store/cache.rs"]
pub(crate) mod cache_mod;
#[path = "store/conversion.rs"]
pub(crate) mod conversion;
#[path = "store/index.rs"]
pub(crate) mod index;
#[path = "store/pool.rs"]
pub(crate) mod pool_mod;

// Re-export the StarStore type and related items from sibling modules.
// Note: BulkInsertState and MemoryMappedState are crate-private; only StarStore
// itself is exposed publicly here, matching the original surface.
pub use crate::store_core::StarStore;
pub use crate::store_indexing::DetailedStorageStatistics;
pub use crate::store_query::StreamingTripleIterator;

// Re-export public types from internal store submodules
pub use bulk_insert_mod::BulkInsertConfig as PublicBulkInsertConfig;
pub use cache_mod::{
    CacheConfig as PublicCacheConfig, CacheStatistics as PublicCacheStatistics,
    StarCache as PublicStarCache,
};
pub use index::IndexStatistics as PublicIndexStatistics;
pub use pool_mod::{
    ConnectionPool as PublicConnectionPool, PoolStatistics as PublicPoolStatistics,
    PooledConnection as PublicPooledConnection,
};
