//! Core-level caching modules for OxiRS.
//!
//! Provides:
//! - [`result_cache`]: Thread-safe SPARQL query result cache with LRU eviction and delta invalidation.
//! - [`triple_cache`]: Generic LRU/LFU/FIFO/TTL cache for RDF triple data, query results, and prefix lookups.

pub mod result_cache;
pub mod triple_cache;

pub use result_cache::{CoreCacheEntry, CoreCacheKey, CoreResultCache};
pub use triple_cache::{
    CachePolicy, CacheStats, PrefixCache, QueryCacheEntry, QueryResultCache, SparqlRow, TripleCache,
};
