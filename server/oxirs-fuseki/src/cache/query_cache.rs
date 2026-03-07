//! Hash-keyed Query Result Cache with Dataset Version Tracking
//!
//! Provides:
//! - `QueryResultCache`: LRU-based cache keyed by `(query_hash, dataset_version)`
//! - `DatasetVersionTracker`: monotonic version counter per dataset for invalidation
//!
//! Design goals
//! - Zero `unwrap()` usage
//! - Thread-safe via `Arc<Mutex<...>>`
//! - TTL-aware: expired entries are evicted on access and by explicit sweep
//! - LRU eviction when the capacity ceiling is reached

use crate::error::{FusekiError, FusekiResult};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A query result binding row — maps variable name → value string.
pub type Binding = HashMap<String, String>;

/// Composite cache key: combines a query hash (e.g. `DefaultHasher` over the
/// normalized SPARQL string) with the dataset's current version counter.
///
/// When the dataset is mutated via SPARQL UPDATE, `DatasetVersionTracker`
/// increments the version, making all previously stored keys stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    /// 64-bit hash of the normalized SPARQL query string
    pub query_hash: u64,
    /// Dataset version at the time the query was cached
    pub dataset_version: u64,
}

impl QueryCacheKey {
    /// Convenience constructor
    pub fn new(query_hash: u64, dataset_version: u64) -> Self {
        QueryCacheKey {
            query_hash,
            dataset_version,
        }
    }
}

/// A single cached query result entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The result rows returned by the SPARQL query
    pub results: Vec<Binding>,
    /// When this entry was inserted
    pub cached_at: Instant,
    /// Time-to-live in milliseconds
    pub ttl_ms: u64,
    /// Number of times this entry was served from cache
    pub hit_count: u64,
}

impl CacheEntry {
    fn new(results: Vec<Binding>, ttl_ms: u64) -> Self {
        CacheEntry {
            results,
            cached_at: Instant::now(),
            ttl_ms,
            hit_count: 0,
        }
    }

    /// Returns `true` if the TTL for this entry has elapsed.
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed().as_millis() as u64 >= self.ttl_ms
    }
}

/// Snapshot statistics for the cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Cumulative cache hits
    pub hits: u64,
    /// Cumulative cache misses
    pub misses: u64,
    /// Cumulative LRU evictions
    pub evictions: u64,
    /// Current number of entries in the cache
    pub size: usize,
}

impl CacheStats {
    /// Hit rate in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QueryResultCache internals
// ──────────────────────────────────────────────────────────────────────────────

struct CacheInner {
    lru: LruCache<QueryCacheKey, CacheEntry>,
    /// Index: dataset_version → set of keys (for bulk invalidation)
    version_index: HashMap<u64, Vec<QueryCacheKey>>,
    evictions: u64,
}

impl CacheInner {
    fn new(capacity: NonZeroUsize) -> Self {
        CacheInner {
            lru: LruCache::new(capacity),
            version_index: HashMap::new(),
            evictions: 0,
        }
    }

    /// Insert an entry, updating the version index and tracking LRU evictions.
    fn insert(&mut self, key: QueryCacheKey, entry: CacheEntry) {
        // Register in version index
        self.version_index
            .entry(key.dataset_version)
            .or_default()
            .push(key);

        // LRU insert; if an old entry is evicted track it
        if let Some((evicted_key, _)) = self.lru.push(key, entry) {
            self.evictions += 1;
            // Clean up the version index for the evicted key
            if let Some(vec) = self.version_index.get_mut(&evicted_key.dataset_version) {
                vec.retain(|k| k != &evicted_key);
            }
        }
    }

    /// Remove a single entry and clean up the version index.
    fn remove(&mut self, key: &QueryCacheKey) {
        self.lru.pop(key);
        if let Some(vec) = self.version_index.get_mut(&key.dataset_version) {
            vec.retain(|k| k != key);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QueryResultCache
// ──────────────────────────────────────────────────────────────────────────────

/// LRU-based cache for SPARQL query results, keyed by `(query_hash, dataset_version)`.
///
/// Thread-safe: all public methods take `&self`.
pub struct QueryResultCache {
    inner: Arc<Mutex<CacheInner>>,
    default_ttl_ms: u64,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl QueryResultCache {
    /// Create a new cache.
    ///
    /// * `capacity`        — maximum number of entries (LRU eviction beyond this)
    /// * `default_ttl_ms`  — default TTL in milliseconds used by `put`
    pub fn new(capacity: usize, default_ttl_ms: u64) -> FusekiResult<Self> {
        let cap = NonZeroUsize::new(capacity).ok_or_else(|| FusekiError::Configuration {
            message: "QueryResultCache capacity must be > 0".to_string(),
        })?;
        Ok(QueryResultCache {
            inner: Arc::new(Mutex::new(CacheInner::new(cap))),
            default_ttl_ms,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Look up a cache entry.
    ///
    /// Returns `None` on miss or if the entry has expired (and removes it).
    pub fn get(&self, key: &QueryCacheKey) -> Option<Vec<Binding>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| {
                tracing::error!("QueryResultCache lock poisoned on get: {}", e);
                e
            })
            .ok()?;

        // Peek first to check expiry without promoting in LRU
        if let Some(entry) = inner.lru.peek(key) {
            if entry.is_expired() {
                let key_clone = *key;
                inner.remove(&key_clone);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }

        // Promote (update LRU order) and return a clone
        if let Some(entry) = inner.lru.get_mut(key) {
            entry.hit_count += 1;
            let results = entry.results.clone();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(results)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store a query result.
    ///
    /// * `key`     — composite cache key
    /// * `results` — the binding rows to cache
    /// * `ttl_ms`  — time-to-live in milliseconds for this entry;
    ///   use `put_default_ttl` to apply the cache's default TTL
    pub fn put(&self, key: QueryCacheKey, results: Vec<Binding>, ttl_ms: u64) {
        let entry = CacheEntry::new(results, ttl_ms);
        match self.inner.lock() {
            Ok(mut inner) => inner.insert(key, entry),
            Err(e) => tracing::error!("QueryResultCache lock poisoned on put: {}", e),
        }
    }

    /// Convenience: `put` using the default TTL configured at construction.
    pub fn put_default_ttl(&self, key: QueryCacheKey, results: Vec<Binding>) {
        let ttl = self.default_ttl_ms;
        self.put(key, results, ttl);
    }

    /// Invalidate all entries cached under a specific `dataset_version`.
    ///
    /// Call this immediately after a SPARQL UPDATE has incremented the dataset
    /// version, so that stale results from the old version are evicted.
    pub fn invalidate_dataset(&self, dataset_version: u64) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("QueryResultCache lock poisoned on invalidate: {}", e);
                return;
            }
        };

        let keys: Vec<QueryCacheKey> = inner
            .version_index
            .remove(&dataset_version)
            .unwrap_or_default();

        for key in &keys {
            inner.lru.pop(key);
        }
    }

    /// Remove all entries whose TTL has elapsed.
    ///
    /// This is an O(n) sweep — call periodically from a background task.
    pub fn evict_expired(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("QueryResultCache lock poisoned on evict_expired: {}", e);
                return;
            }
        };

        // Collect expired keys without modifying the LruCache while iterating
        let expired: Vec<QueryCacheKey> = inner
            .lru
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| *key)
            .collect();

        for key in &expired {
            inner.remove(key);
        }
    }

    /// Snapshot statistics.
    pub fn stats(&self) -> CacheStats {
        let size = self.inner.lock().map(|inner| inner.lru.len()).unwrap_or(0);

        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.inner.lock().map(|inner| inner.evictions).unwrap_or(0),
            size,
        }
    }

    /// Hit rate in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DatasetVersionTracker
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks a monotonically-increasing version counter per named dataset.
///
/// Each SPARQL UPDATE call should call `increment_version(dataset_id)` to
/// bump the counter.  The new version is then used as the `dataset_version`
/// component of `QueryCacheKey`s for subsequent queries, automatically
/// invalidating all prior cache entries for that dataset.
pub struct DatasetVersionTracker {
    versions: Arc<Mutex<HashMap<String, u64>>>,
}

impl DatasetVersionTracker {
    /// Create a new tracker with no datasets registered.
    pub fn new() -> Self {
        DatasetVersionTracker {
            versions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the current version for `dataset_id`, or `0` if never tracked.
    pub fn get_version(&self, dataset_id: &str) -> u64 {
        self.versions
            .lock()
            .map(|map| *map.get(dataset_id).unwrap_or(&0))
            .unwrap_or(0)
    }

    /// Atomically increment the version for `dataset_id` and return the new value.
    ///
    /// The first increment on a previously-unseen dataset goes from 0 → 1.
    pub fn increment_version(&self, dataset_id: &str) -> u64 {
        match self.versions.lock() {
            Ok(mut map) => {
                let v = map.entry(dataset_id.to_string()).or_insert(0);
                *v = v.saturating_add(1);
                *v
            }
            Err(e) => {
                tracing::error!("DatasetVersionTracker lock poisoned on increment: {}", e);
                0
            }
        }
    }

    /// Reset the version counter for `dataset_id` to 0.
    ///
    /// Use this when a dataset is dropped or re-created from scratch.
    pub fn reset(&self, dataset_id: &str) {
        match self.versions.lock() {
            Ok(mut map) => {
                map.insert(dataset_id.to_string(), 0);
            }
            Err(e) => {
                tracing::error!("DatasetVersionTracker lock poisoned on reset: {}", e);
            }
        }
    }

    /// List all dataset IDs currently tracked.
    pub fn dataset_ids(&self) -> Vec<String> {
        self.versions
            .lock()
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for DatasetVersionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_cache(capacity: usize, ttl_ms: u64) -> QueryResultCache {
        QueryResultCache::new(capacity, ttl_ms).unwrap()
    }

    fn make_key(hash: u64, version: u64) -> QueryCacheKey {
        QueryCacheKey::new(hash, version)
    }

    fn make_bindings(rows: usize) -> Vec<Binding> {
        (0..rows)
            .map(|i| {
                let mut m = Binding::new();
                m.insert("s".to_string(), format!("http://ex.org/s{}", i));
                m
            })
            .collect()
    }

    // ── QueryCacheKey ─────────────────────────────────────────────────────────

    #[test]
    fn test_key_equality() {
        let k1 = make_key(42, 1);
        let k2 = make_key(42, 1);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_key_inequality_hash() {
        let k1 = make_key(42, 1);
        let k2 = make_key(99, 1);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_key_inequality_version() {
        let k1 = make_key(42, 1);
        let k2 = make_key(42, 2);
        assert_ne!(k1, k2);
    }

    // ── CacheEntry expiry ─────────────────────────────────────────────────────

    #[test]
    fn test_entry_not_expired_immediately() {
        let entry = CacheEntry::new(vec![], 60_000);
        assert!(!entry.is_expired(), "Should not be expired immediately");
    }

    #[test]
    fn test_entry_expires_after_ttl() {
        let entry = CacheEntry::new(vec![], 1); // 1 ms TTL
        std::thread::sleep(Duration::from_millis(10));
        assert!(entry.is_expired(), "Should be expired after TTL elapsed");
    }

    // ── QueryResultCache: basic hit/miss ──────────────────────────────────────

    #[test]
    fn test_cache_miss_on_empty() {
        let cache = make_cache(100, 60_000);
        let key = make_key(1, 0);
        assert!(cache.get(&key).is_none(), "Empty cache should miss");
    }

    #[test]
    fn test_cache_put_and_hit() {
        let cache = make_cache(100, 60_000);
        let key = make_key(1, 0);
        let bindings = make_bindings(3);

        cache.put(key, bindings.clone(), 60_000);

        let result = cache.get(&make_key(1, 0));
        assert!(result.is_some(), "Should hit after put");
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_cache_miss_after_ttl_expiry() {
        let cache = make_cache(100, 1); // 1 ms TTL
        let key = make_key(1, 0);
        cache.put(key, make_bindings(1), 1);

        std::thread::sleep(Duration::from_millis(10));

        assert!(
            cache.get(&make_key(1, 0)).is_none(),
            "Should miss after TTL expires"
        );
    }

    #[test]
    fn test_put_default_ttl() {
        let cache = make_cache(100, 60_000);
        let key = make_key(7, 3);
        cache.put_default_ttl(key, make_bindings(2));
        assert!(cache.get(&make_key(7, 3)).is_some());
    }

    // ── QueryResultCache: LRU eviction ────────────────────────────────────────

    #[test]
    fn test_lru_eviction_at_capacity() {
        let cache = make_cache(2, 60_000);

        cache.put(make_key(1, 0), make_bindings(1), 60_000);
        cache.put(make_key(2, 0), make_bindings(1), 60_000);
        // Access key 1 to make key 2 LRU
        let _ = cache.get(&make_key(1, 0));
        // Insert key 3 → should evict key 2 (LRU)
        cache.put(make_key(3, 0), make_bindings(1), 60_000);

        assert!(cache.get(&make_key(1, 0)).is_some(), "key 1 should survive");
        assert!(cache.get(&make_key(3, 0)).is_some(), "key 3 should survive");

        let stats = cache.stats();
        assert!(
            stats.evictions >= 1,
            "Should have at least one LRU eviction"
        );
    }

    // ── QueryResultCache: invalidate_dataset ─────────────────────────────────

    #[test]
    fn test_invalidate_dataset_removes_matching_version() {
        let cache = make_cache(100, 60_000);

        cache.put(make_key(1, 1), make_bindings(1), 60_000);
        cache.put(make_key(2, 1), make_bindings(1), 60_000);
        cache.put(make_key(3, 2), make_bindings(1), 60_000);

        cache.invalidate_dataset(1);

        assert!(
            cache.get(&make_key(1, 1)).is_none(),
            "Version 1, hash 1 should be gone"
        );
        assert!(
            cache.get(&make_key(2, 1)).is_none(),
            "Version 1, hash 2 should be gone"
        );
        assert!(
            cache.get(&make_key(3, 2)).is_some(),
            "Version 2 should survive"
        );
    }

    #[test]
    fn test_invalidate_nonexistent_version_is_noop() {
        let cache = make_cache(100, 60_000);
        cache.put(make_key(1, 0), make_bindings(1), 60_000);
        // Invalidate a version that was never used
        cache.invalidate_dataset(999);
        assert!(cache.get(&make_key(1, 0)).is_some(), "Entry should survive");
    }

    // ── QueryResultCache: evict_expired ──────────────────────────────────────

    #[test]
    fn test_evict_expired_removes_expired_entries() {
        let cache = make_cache(100, 60_000);

        cache.put(make_key(1, 0), make_bindings(1), 1); // expires fast
        cache.put(make_key(2, 0), make_bindings(1), 60_000); // long TTL

        std::thread::sleep(Duration::from_millis(10));

        cache.evict_expired();

        assert!(
            cache.get(&make_key(1, 0)).is_none(),
            "Expired entry should be gone after evict_expired"
        );
        assert!(
            cache.get(&make_key(2, 0)).is_some(),
            "Non-expired entry should remain"
        );
    }

    // ── QueryResultCache: stats ───────────────────────────────────────────────

    #[test]
    fn test_stats_hit_and_miss_counts() {
        let cache = make_cache(100, 60_000);
        let key = make_key(1, 0);
        cache.put(key, make_bindings(1), 60_000);

        cache.get(&make_key(1, 0)); // hit
        cache.get(&make_key(99, 0)); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_stats_size_matches_entries() {
        let cache = make_cache(100, 60_000);
        assert_eq!(cache.stats().size, 0);

        cache.put(make_key(1, 0), make_bindings(1), 60_000);
        assert_eq!(cache.stats().size, 1);

        cache.put(make_key(2, 0), make_bindings(1), 60_000);
        assert_eq!(cache.stats().size, 2);
    }

    // ── QueryResultCache: hit_rate ────────────────────────────────────────────

    #[test]
    fn test_hit_rate_zero_on_empty() {
        let cache = make_cache(100, 60_000);
        assert!((cache.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_one_hit_one_miss() {
        let cache = make_cache(100, 60_000);
        cache.put(make_key(1, 0), make_bindings(1), 60_000);

        cache.get(&make_key(1, 0)); // hit
        cache.get(&make_key(2, 0)); // miss

        assert!(
            (cache.hit_rate() - 0.5).abs() < f64::EPSILON,
            "Expected 0.5, got {}",
            cache.hit_rate()
        );
    }

    #[test]
    fn test_hit_rate_all_misses() {
        let cache = make_cache(100, 60_000);
        cache.get(&make_key(1, 0));
        cache.get(&make_key(2, 0));
        assert!((cache.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    // ── QueryResultCache: capacity=0 should fail ─────────────────────────────

    #[test]
    fn test_zero_capacity_returns_error() {
        let result = QueryResultCache::new(0, 60_000);
        assert!(result.is_err(), "Zero capacity should fail");
    }

    // ── DatasetVersionTracker ────────────────────────────────────────────────

    #[test]
    fn test_version_tracker_initial_zero() {
        let tracker = DatasetVersionTracker::new();
        assert_eq!(tracker.get_version("ds1"), 0);
    }

    #[test]
    fn test_version_tracker_increment() {
        let tracker = DatasetVersionTracker::new();
        let v1 = tracker.increment_version("ds1");
        assert_eq!(v1, 1, "First increment should return 1");

        let v2 = tracker.increment_version("ds1");
        assert_eq!(v2, 2, "Second increment should return 2");
    }

    #[test]
    fn test_version_tracker_reset() {
        let tracker = DatasetVersionTracker::new();
        tracker.increment_version("ds1");
        tracker.increment_version("ds1");
        tracker.reset("ds1");
        assert_eq!(tracker.get_version("ds1"), 0, "After reset, version is 0");
    }

    #[test]
    fn test_version_tracker_multiple_datasets() {
        let tracker = DatasetVersionTracker::new();
        tracker.increment_version("ds1");
        tracker.increment_version("ds1");
        tracker.increment_version("ds2");

        assert_eq!(tracker.get_version("ds1"), 2);
        assert_eq!(tracker.get_version("ds2"), 1);
        assert_eq!(tracker.get_version("ds3"), 0, "Unknown dataset returns 0");
    }

    #[test]
    fn test_version_tracker_get_after_increment() {
        let tracker = DatasetVersionTracker::new();
        let v = tracker.increment_version("myds");
        assert_eq!(tracker.get_version("myds"), v);
    }

    #[test]
    fn test_version_tracker_dataset_ids() {
        let tracker = DatasetVersionTracker::new();
        tracker.increment_version("a");
        tracker.increment_version("b");
        tracker.increment_version("c");

        let mut ids = tracker.dataset_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    // ── CacheStats helpers ────────────────────────────────────────────────────

    #[test]
    fn test_cache_stats_hit_rate_method() {
        let stats = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
            size: 3,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero_total() {
        let stats = CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            size: 0,
        };
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }
}
