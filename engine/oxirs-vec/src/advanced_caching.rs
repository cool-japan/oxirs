//! Advanced multi-level caching system for vector embeddings and search results.
//!
//! This module is organized as a set of focused sub-modules:
//!
//! - [`advanced_caching_eviction`] — `MemoryCache`, `PersistentCache`, eviction policies
//! - [`advanced_caching_multilevel`] — `MultiLevelCache`, `CacheInvalidator`
//! - [`advanced_caching_worker`] — `BackgroundCacheWorker`, `CacheWarmer`, `CacheAnalyzer`
//!
//! Shared primitive types (`EvictionPolicy`, `CacheConfig`, `CacheEntry`, `CacheKey`,
//! `CacheStats`) are defined here in the root and re-exported so that all sibling
//! modules can import them via `use super::*` or specific `use super::{…}` paths.

use crate::Vector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Re-exports from sibling modules (declared as top-level modules in lib.rs)
// ---------------------------------------------------------------------------

// The three sibling files live flat in src/ and are declared as top-level
// modules in lib.rs.  We re-export their public types here so that the
// `advanced_caching::*` public surface is unchanged for callers.

pub use crate::advanced_caching_eviction::{MemoryCache, PersistentCache};
pub use crate::advanced_caching_multilevel::{
    CacheInvalidator, InvalidationStats, MultiLevelCache, MultiLevelCacheStats,
};
pub use crate::advanced_caching_worker::{
    BackgroundCacheWorker, CacheAnalysisReport, CacheAnalyzer, CacheWarmer,
};

// ---------------------------------------------------------------------------
// Shared primitive types (used by all sub-modules via `use super::*`)
// ---------------------------------------------------------------------------

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive Replacement Cache
    ARC,
    /// First In, First Out
    FIFO,
    /// Time-based expiration only
    TTL,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in memory cache
    pub max_memory_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Time-to-live for cache entries
    pub ttl: Option<Duration>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable persistent cache
    pub enable_persistent: bool,
    /// Persistent cache directory
    pub persistent_cache_dir: Option<std::path::PathBuf>,
    /// Maximum persistent cache size in bytes
    pub max_persistent_bytes: usize,
    /// Enable cache compression
    pub enable_compression: bool,
    /// Enable background updates
    pub enable_background_updates: bool,
    /// Background update interval
    pub background_update_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10_000,
            max_memory_bytes: 1024 * 1024 * 100,  // 100 MB
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            enable_persistent: true,
            persistent_cache_dir: None,
            max_persistent_bytes: 1024 * 1024 * 1024, // 1 GB
            enable_compression: true,
            enable_background_updates: false,
            background_update_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached data
    pub data: Vector,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access count for LFU
    pub access_count: u64,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// TTL for this specific entry
    pub ttl: Option<Duration>,
    /// Metadata tags
    pub tags: HashMap<String, String>,
}

impl CacheEntry {
    pub fn new(data: Vector) -> Self {
        let now = Instant::now();
        let size_bytes = data.dimensions * std::mem::size_of::<f32>() + 64;

        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            ttl: None,
            tags: HashMap::new(),
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }

    /// Returns `true` if this entry's TTL has elapsed.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Update access statistics.
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Cache key that can be hashed
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub namespace: String,
    pub key: String,
    pub variant: Option<String>,
}

impl CacheKey {
    pub fn new(namespace: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            variant: None,
        }
    }

    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref variant) = self.variant {
            write!(f, "{}:{}:{}", self.namespace, self.key, variant)
        } else {
            write!(f, "{}:{}", self.namespace, self.key)
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries: usize,
    pub memory_bytes: usize,
    pub max_entries: usize,
    pub max_memory_bytes: usize,
    pub hit_ratio: f32,
}
