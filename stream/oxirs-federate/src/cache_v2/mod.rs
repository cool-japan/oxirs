//! Advanced multi-level cache with dependency tracking for federated queries.
//!
//! This module provides:
//! - [`MultiLevelCache`] – L1 (in-memory LRU) + L2 (disk-backed) cache.
//! - [`DependencyTracker`] – DAG-based dependency graph for smart invalidation.
//! - [`CacheKey`] / [`CacheEntry`] – shared cache types.

pub mod dependency_tracker;
pub mod multi_level_cache;

pub use dependency_tracker::{DependencyKind, DependencyTracker};
pub use multi_level_cache::{
    CacheEntry, CacheKey, DiskCache, MemoryCache, MultiLevelCache, MultiLevelCacheStats,
};
