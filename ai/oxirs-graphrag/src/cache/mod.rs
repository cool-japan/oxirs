//! Cache module for GraphRAG query results
//!
//! Provides a thread-safe LRU cache with TTL-based expiry.

pub mod query_cache;

pub use query_cache::{CacheEntry, CacheStats, QueryCache, QueryCacheConfig};
