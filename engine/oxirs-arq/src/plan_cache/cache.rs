//! Bounded LRU plan cache with schema-version invalidation.
//!
//! [`PlanCache<V>`] is a thread-safe, bounded cache keyed by `u64` fingerprints
//! (produced by [`crate::plan_cache::fingerprint::compute_fingerprint`]).  It
//! uses [`parking_lot::RwLock`] for low-contention concurrent reads and an
//! [`LruEviction`] to bound memory usage.
//!
//! ## Schema invalidation
//! Callers can call [`PlanCache::invalidate_all`] when the schema changes (e.g.
//! after a graph update that invalidates cardinality assumptions).  This bumps
//! an internal `schema_version` counter and clears all entries.
//!
//! ## Thread safety
//! [`PlanCache`] wraps its internals in `Arc<RwLock<…>>` and implements
//! [`Clone`]; clones share the same backing store and therefore observe the
//! same hits/misses/evictions.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::eviction::LruEviction;

/// Hit/miss/eviction counters for a [`PlanCache`].
///
/// Returned by [`PlanCache::stats`] as a plain `(hits, misses, evictions)` tuple.
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of failed cache lookups.
    pub misses: u64,
    /// Number of entries evicted due to capacity pressure.
    pub evictions: u64,
}

struct CacheInner<V> {
    map: HashMap<u64, V>,
    eviction: LruEviction,
    stats: CacheStats,
    /// Bumped by [`PlanCache::invalidate_all`].  Not currently stored per-entry
    /// (full invalidation clears the map), but tracked for observability.
    schema_version: u64,
}

/// A bounded LRU plan cache with schema-version invalidation.
///
/// `V` is typically [`crate::algebra::Algebra`] (the optimised plan) but the
/// cache is generic so tests can use `String`, `u32`, etc.
///
/// ```rust
/// use oxirs_arq::plan_cache::PlanCache;
///
/// let cache: PlanCache<String> = PlanCache::new(10);
/// cache.insert(42, "plan-a".to_string());
/// assert_eq!(cache.get(42).as_deref(), Some("plan-a"));
///
/// let (hits, misses, evictions) = cache.stats();
/// assert_eq!(hits, 1);
/// assert_eq!(misses, 0);
/// assert_eq!(evictions, 0);
/// ```
pub struct PlanCache<V: Clone> {
    inner: Arc<RwLock<CacheInner<V>>>,
}

impl<V: Clone> PlanCache<V> {
    /// Create a new cache with the given maximum `capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                map: HashMap::new(),
                eviction: LruEviction::new(capacity),
                stats: CacheStats {
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                },
                schema_version: 0,
            })),
        }
    }

    /// Look up `key`.  Returns a clone of the stored value on a hit, or `None`
    /// on a miss.  Updates hit/miss counters and refreshes the LRU position.
    pub fn get(&self, key: u64) -> Option<V> {
        let mut inner = self.inner.write();
        if let Some(val) = inner.map.get(&key).cloned() {
            inner.eviction.on_access(key);
            inner.stats.hits += 1;
            Some(val)
        } else {
            inner.stats.misses += 1;
            None
        }
    }

    /// Insert `value` under `key`.  If the cache is at capacity, the
    /// least-recently-used entry is evicted first.
    pub fn insert(&self, key: u64, value: V) {
        let mut inner = self.inner.write();
        if let Some(evict_key) = inner.eviction.on_insert(key) {
            inner.map.remove(&evict_key);
            inner.stats.evictions += 1;
        }
        inner.map.insert(key, value);
    }

    /// Remove all entries and bump the schema version.
    pub fn invalidate_all(&self) {
        let mut inner = self.inner.write();
        inner.map.clear();
        // Reset eviction tracker to avoid stale ordering.
        inner.eviction = LruEviction::new(inner.eviction.capacity());
        inner.schema_version += 1;
    }

    /// Return `(hits, misses, evictions)` since construction.
    pub fn stats(&self) -> (u64, u64, u64) {
        let inner = self.inner.read();
        (inner.stats.hits, inner.stats.misses, inner.stats.evictions)
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.read().map.len()
    }

    /// Returns `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.read().map.is_empty()
    }

    /// Current schema version (bumped by each [`invalidate_all`](Self::invalidate_all) call).
    pub fn schema_version(&self) -> u64 {
        self.inner.read().schema_version
    }
}

impl<V: Clone> Clone for PlanCache<V> {
    /// Clone returns a handle that shares the **same** backing store.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_and_miss_counters() {
        let cache: PlanCache<String> = PlanCache::new(10);
        cache.insert(1, "a".into());
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn capacity_evicts_lru() {
        let cache: PlanCache<u32> = PlanCache::new(3);
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);
        cache.insert(4, 40); // evicts 1
        assert!(cache.get(1).is_none(), "key 1 should be evicted");
        assert_eq!(cache.get(4), Some(40));
    }

    #[test]
    fn invalidate_all_clears_and_bumps_version() {
        let cache: PlanCache<u32> = PlanCache::new(10);
        cache.insert(1, 100);
        assert_eq!(cache.schema_version(), 0);
        cache.invalidate_all();
        assert!(cache.get(1).is_none());
        assert_eq!(cache.schema_version(), 1);
    }

    #[test]
    fn clone_shares_backing_store() {
        let cache: PlanCache<u32> = PlanCache::new(10);
        let clone = cache.clone();
        cache.insert(99, 42);
        assert_eq!(clone.get(99), Some(42));
    }

    #[test]
    fn is_empty_initially() {
        let cache: PlanCache<u32> = PlanCache::new(10);
        assert!(cache.is_empty());
        cache.insert(1, 1);
        assert!(!cache.is_empty());
    }
}
