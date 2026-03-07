//! GraphQL Query Result Caching
//!
//! This module provides an in-process, schema-aware query result cache that
//! stores serialised GraphQL responses and supports invalidation based on
//! the RDF graphs that were accessed when producing each response.

pub mod invalidation;
pub mod query_cache;
pub mod response_cache;

pub use invalidation::{
    CacheInvalidationManager, InvalidationAudit, InvalidationEvent, InvalidationResult,
    InvalidationRule, PredicateInvalidationIndex,
};
pub use query_cache::{CacheEntry, CacheKey, CacheStats, GqlQueryCache};
pub use response_cache::{
    CachePolicy, CacheScope, CachedResponse, FieldLevelCacheDirective, GraphQlResponseCache,
    ResponseCacheKey,
};
