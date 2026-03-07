//! Caching subsystem for OxiRS Fuseki
//!
//! Provides intelligent query result caching with semantic invalidation.

pub mod query_cache;
pub mod sparql_cache;

pub use query_cache::{
    Binding, CacheEntry, CacheStats as QueryCacheStats, DatasetVersionTracker,
    QueryCacheKey as QueryResultCacheKey, QueryResultCache,
};
pub use sparql_cache::{
    CacheStats, CachedQueryResult, QueryCacheKey, SparqlQueryCache, SparqlQueryType,
};
