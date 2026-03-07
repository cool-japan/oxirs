//! Federated query result cache with per-endpoint invalidation.
//!
//! Provides [`FederatedQueryCache`], a thread-safe LRU cache that tracks
//! which SPARQL endpoints contributed to each cached result.  When an
//! endpoint's data changes, all affected entries can be invalidated atomically.

#[allow(clippy::module_inception)]
pub mod federated_cache;

pub use federated_cache::{
    FederatedCacheEntry, FederatedCacheKey, FederatedCacheStats, FederatedQueryCache,
};
