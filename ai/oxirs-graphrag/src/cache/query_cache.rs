//! Thread-safe LRU cache with TTL-based expiry for GraphRAG query results.
//!
//! # Design
//!
//! This cache implements the Least Recently Used (LRU) eviction policy combined
//! with Time-To-Live (TTL) expiry. Each cache entry carries a timestamp and TTL
//! value; entries that have exceeded their TTL are treated as stale and evicted
//! lazily on the next access.
//!
//! ## Thread Safety
//!
//! The cache is wrapped in `Arc<Mutex<...>>` and exposes only `&self` methods
//! so it can be shared across threads and async tasks without additional
//! synchronization from the caller.
//!
//! ## Complexity
//!
//! | Operation | Amortized | Worst-Case |
//! |-----------|-----------|------------|
//! | `get`     | O(1)      | O(1)       |
//! | `put`     | O(1)      | O(1)       |
//! | `remove`  | O(1)      | O(1)       |
//! | `evict_expired` | O(n) | O(n)     |
//!
//! The underlying doubly-linked-list + hash-map structure comes from the
//! `lru` crate.

use lru::LruCache;
use std::{
    hash::Hash,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{GraphRAGError, GraphRAGResult};

/// Configuration for [`QueryCache`].
#[derive(Debug, Clone)]
pub struct QueryCacheConfig {
    /// Maximum number of entries the cache will hold.
    ///
    /// When capacity is reached, the least-recently-used entry is evicted first,
    /// unless there are stale (TTL-expired) entries which are evicted before LRU.
    pub capacity: NonZeroUsize,

    /// Default TTL for entries that do not specify their own TTL.
    pub default_ttl: Duration,

    /// Minimum allowed TTL (lower bound for per-entry TTL).
    pub min_ttl: Duration,

    /// Maximum allowed TTL (upper bound for per-entry TTL).
    pub max_ttl: Duration,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            capacity: NonZeroUsize::new(1024).expect("1024 is non-zero"),
            default_ttl: Duration::from_secs(3600), // 1 hour
            min_ttl: Duration::from_secs(300),      // 5 minutes
            max_ttl: Duration::from_secs(86_400),   // 24 hours
        }
    }
}

/// A single cached value together with its metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value.
    pub value: V,
    /// Wall-clock instant when this entry was inserted.
    pub inserted_at: Instant,
    /// How long the entry lives before it is considered stale.
    pub ttl: Duration,
    /// How many times this entry has been read since insertion.
    pub hit_count: u64,
}

impl<V: Clone> CacheEntry<V> {
    /// Returns `true` if the entry has not yet exceeded its TTL.
    #[inline]
    pub fn is_fresh(&self) -> bool {
        self.inserted_at.elapsed() < self.ttl
    }

    /// Remaining lifetime. Returns [`Duration::ZERO`] if the entry has expired.
    #[inline]
    pub fn remaining_ttl(&self) -> Duration {
        let elapsed = self.inserted_at.elapsed();
        self.ttl.saturating_sub(elapsed)
    }
}

/// Snapshot of cache performance metrics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of cache hits (fresh entry found).
    pub hits: u64,
    /// Total number of cache misses (entry absent or stale).
    pub misses: u64,
    /// Total number of stale entries that were evicted.
    pub stale_evictions: u64,
    /// Total number of capacity-driven LRU evictions.
    pub lru_evictions: u64,
    /// Current number of live (non-stale) entries.
    pub live_entries: usize,
    /// Maximum capacity of the cache.
    pub capacity: usize,
}

impl CacheStats {
    /// Hit rate in the range `[0.0, 1.0]`. Returns `0.0` if no lookups yet.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ─── Inner state ─────────────────────────────────────────────────────────────

struct CacheInner<K, V> {
    lru: LruCache<K, CacheEntry<V>>,
    stats: CacheStats,
    config: QueryCacheConfig,
}

impl<K: Hash + Eq + Clone, V: Clone> CacheInner<K, V> {
    fn new(config: QueryCacheConfig) -> Self {
        let capacity = config.capacity;
        Self {
            lru: LruCache::new(capacity),
            stats: CacheStats {
                capacity: capacity.get(),
                ..Default::default()
            },
            config,
        }
    }

    /// Clamp a user-supplied TTL to `[min_ttl, max_ttl]`.
    fn clamp_ttl(&self, ttl: Duration) -> Duration {
        ttl.max(self.config.min_ttl).min(self.config.max_ttl)
    }

    /// Look up a key.  
    /// Returns a *clone* of the value if found and fresh.  
    /// Stale entries are removed from the LRU and counted as misses.
    fn get(&mut self, key: &K) -> Option<V> {
        // `peek` to check freshness without promoting to MRU yet.
        let is_stale = match self.lru.peek(key) {
            Some(entry) => !entry.is_fresh(),
            None => {
                self.stats.misses += 1;
                return None;
            }
        };

        if is_stale {
            self.lru.pop(key);
            self.stats.stale_evictions += 1;
            self.stats.misses += 1;
            self.stats.live_entries = self.lru.len();
            return None;
        }

        // Promote to MRU and return a clone.
        if let Some(entry) = self.lru.get_mut(key) {
            entry.hit_count += 1;
            let value = entry.value.clone();
            self.stats.hits += 1;
            Some(value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert or replace a cache entry with a specific TTL.
    fn put_with_ttl(&mut self, key: K, value: V, ttl: Duration) {
        let ttl = self.clamp_ttl(ttl);

        // If at capacity, track LRU eviction before inserting.
        if self.lru.len() == self.lru.cap().get() {
            // Check if the candidate for eviction is stale; if so it is a
            // stale_eviction rather than an LRU eviction.
            let oldest_stale = self
                .lru
                .peek_lru()
                .map(|(_, e)| !e.is_fresh())
                .unwrap_or(false);
            if oldest_stale {
                self.stats.stale_evictions += 1;
            } else {
                self.stats.lru_evictions += 1;
            }
        }

        let entry = CacheEntry {
            value,
            inserted_at: Instant::now(),
            ttl,
            hit_count: 0,
        };
        self.lru.put(key, entry);
        self.stats.live_entries = self.lru.len();
    }

    /// Insert or replace a cache entry using the default TTL from config.
    fn put(&mut self, key: K, value: V) {
        let ttl = self.config.default_ttl;
        self.put_with_ttl(key, value, ttl);
    }

    /// Remove an entry by key. Returns the value if it was present and fresh.
    fn remove(&mut self, key: &K) -> Option<V> {
        let entry = self.lru.pop(key)?;
        self.stats.live_entries = self.lru.len();
        if entry.is_fresh() {
            Some(entry.value)
        } else {
            self.stats.stale_evictions += 1;
            None
        }
    }

    /// Scan all entries and remove stale ones. Returns the number of evictions.
    fn evict_expired(&mut self) -> usize {
        let stale_keys: Vec<K> = self
            .lru
            .iter()
            .filter(|(_, entry)| !entry.is_fresh())
            .map(|(k, _)| k.clone())
            .collect();

        let count = stale_keys.len();
        for key in stale_keys {
            self.lru.pop(&key);
        }

        self.stats.stale_evictions += count as u64;
        self.stats.live_entries = self.lru.len();
        count
    }

    /// Inspect a cache entry without affecting LRU order.
    fn peek_entry(&self, key: &K) -> Option<&CacheEntry<V>> {
        self.lru.peek(key)
    }

    /// Return a snapshot of current stats.
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clear the cache entirely.
    fn clear(&mut self) {
        self.lru.clear();
        self.stats.live_entries = 0;
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// A thread-safe LRU cache with per-entry TTL expiry.
///
/// # Type Parameters
///
/// * `K` – Key type. Must be `Hash + Eq + Clone + Send + Sync`.
/// * `V` – Value type. Must be `Clone + Send + Sync`.
///
/// # Example
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
/// use oxirs_graphrag::cache::query_cache::{QueryCache, QueryCacheConfig};
///
/// let config = QueryCacheConfig {
///     capacity: NonZeroUsize::new(100).expect("should succeed"),
///     default_ttl: Duration::from_secs(60),
///     ..Default::default()
/// };
/// let cache: QueryCache<String, String> = QueryCache::new(config);
///
/// cache.put("hello".to_string(), "world".to_string()).expect("should succeed");
/// assert_eq!(cache.get(&"hello".to_string()).expect("should succeed"), Some("world".to_string()));
/// ```
#[derive(Clone)]
pub struct QueryCache<K, V> {
    inner: Arc<Mutex<CacheInner<K, V>>>,
}

impl<K, V> std::fmt::Debug for QueryCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCache").finish_non_exhaustive()
    }
}

impl<K, V> QueryCache<K, V>
where
    K: Hash + Eq + Clone + Send + 'static,
    V: Clone + Send + 'static,
{
    /// Create a new `QueryCache` from the given configuration.
    pub fn new(config: QueryCacheConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner::new(config))),
        }
    }

    /// Create a `QueryCache` with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(QueryCacheConfig::default())
    }

    /// Look up `key` and return the value if present and fresh.
    ///
    /// Stale entries are evicted; their keys count as misses.
    pub fn get(&self, key: &K) -> GraphRAGResult<Option<V>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.get(key))
    }

    /// Insert `value` under `key` using the cache's default TTL.
    pub fn put(&self, key: K, value: V) -> GraphRAGResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        guard.put(key, value);
        Ok(())
    }

    /// Insert `value` under `key` with an explicit TTL.
    ///
    /// The TTL is clamped to `[min_ttl, max_ttl]` from the config.
    pub fn put_with_ttl(&self, key: K, value: V, ttl: Duration) -> GraphRAGResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        guard.put_with_ttl(key, value, ttl);
        Ok(())
    }

    /// Remove an entry. Returns the value if it existed and was fresh.
    pub fn remove(&self, key: &K) -> GraphRAGResult<Option<V>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.remove(key))
    }

    /// Evict all stale entries and return the count of evictions.
    ///
    /// This is O(n) in the number of entries. Call it from a periodic
    /// maintenance task to keep memory usage bounded.
    pub fn evict_expired(&self) -> GraphRAGResult<usize> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.evict_expired())
    }

    /// Return a snapshot of cache performance statistics.
    pub fn stats(&self) -> GraphRAGResult<CacheStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.stats())
    }

    /// Peek at an entry's metadata without affecting LRU order.
    ///
    /// Returns `None` if the entry is absent or stale.
    pub fn peek_entry<F, R>(&self, key: &K, f: F) -> GraphRAGResult<Option<R>>
    where
        F: FnOnce(&CacheEntry<V>) -> R,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        match guard.peek_entry(key) {
            Some(entry) if entry.is_fresh() => Ok(Some(f(entry))),
            _ => Ok(None),
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) -> GraphRAGResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        guard.clear();
        Ok(())
    }

    /// Return the current number of live (non-stale) entries.
    ///
    /// Note: this includes any entries that *might* have expired between
    /// insert time and this call but have not yet been lazily evicted.
    pub fn len(&self) -> GraphRAGResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.lru.len())
    }

    /// Return `true` if the cache contains no entries.
    pub fn is_empty(&self) -> GraphRAGResult<bool> {
        Ok(self.len()? == 0)
    }

    /// Return the maximum capacity of the cache.
    pub fn capacity(&self) -> GraphRAGResult<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| GraphRAGError::InternalError("cache mutex poisoned".to_string()))?;
        Ok(guard.lru.cap().get())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn small_cache(cap: usize, ttl_secs: u64) -> QueryCache<String, String> {
        QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(cap).expect("cap is non-zero"),
            default_ttl: Duration::from_secs(ttl_secs),
            min_ttl: Duration::from_millis(1),
            max_ttl: Duration::from_secs(86_400),
        })
    }

    #[test]
    fn test_basic_put_get() {
        let cache = small_cache(10, 3600);
        cache
            .put("key1".to_string(), "value1".to_string())
            .expect("should succeed");
        let result = cache.get(&"key1".to_string()).expect("should succeed");
        assert_eq!(result, Some("value1".to_string()));
    }

    #[test]
    fn test_miss_on_absent_key() {
        let cache: QueryCache<String, String> = small_cache(10, 3600);
        let result = cache.get(&"absent".to_string()).expect("should succeed");
        assert_eq!(result, None);
    }

    #[test]
    fn test_overwrite_key() {
        let cache = small_cache(10, 3600);
        cache
            .put("k".to_string(), "v1".to_string())
            .expect("should succeed");
        cache
            .put("k".to_string(), "v2".to_string())
            .expect("should succeed");
        let result = cache.get(&"k".to_string()).expect("should succeed");
        assert_eq!(result, Some("v2".to_string()));
    }

    #[test]
    fn test_ttl_expiry() {
        let cache = QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(10).expect("should succeed"),
            default_ttl: Duration::from_millis(50),
            min_ttl: Duration::from_millis(1),
            max_ttl: Duration::from_secs(3600),
        });
        cache
            .put("k".to_string(), "v".to_string())
            .expect("should succeed");
        // Entry should be fresh immediately.
        assert_eq!(
            cache.get(&"k".to_string()).expect("should succeed"),
            Some("v".to_string())
        );
        // Wait for TTL to expire.
        thread::sleep(Duration::from_millis(100));
        // Entry should now be stale / evicted.
        assert_eq!(cache.get(&"k".to_string()).expect("should succeed"), None);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = small_cache(3, 3600);
        cache
            .put("a".to_string(), "1".to_string())
            .expect("should succeed");
        cache
            .put("b".to_string(), "2".to_string())
            .expect("should succeed");
        cache
            .put("c".to_string(), "3".to_string())
            .expect("should succeed");

        // Access "a" to make it recently used; "b" becomes LRU.
        let _ = cache.get(&"a".to_string()).expect("should succeed");

        // Insert "d" – should evict "b" (LRU).
        cache
            .put("d".to_string(), "4".to_string())
            .expect("should succeed");

        assert_eq!(
            cache.get(&"b".to_string()).expect("should succeed"),
            None,
            "b should be evicted"
        );
        assert!(
            cache
                .get(&"a".to_string())
                .expect("should succeed")
                .is_some(),
            "a should survive"
        );
        assert!(
            cache
                .get(&"d".to_string())
                .expect("should succeed")
                .is_some(),
            "d should be present"
        );
    }

    #[test]
    fn test_remove() {
        let cache = small_cache(10, 3600);
        cache
            .put("k".to_string(), "v".to_string())
            .expect("should succeed");
        let removed = cache.remove(&"k".to_string()).expect("should succeed");
        assert_eq!(removed, Some("v".to_string()));
        assert_eq!(cache.get(&"k".to_string()).expect("should succeed"), None);
    }

    #[test]
    fn test_evict_expired_batch() {
        let cache = QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(20).expect("should succeed"),
            default_ttl: Duration::from_millis(50),
            min_ttl: Duration::from_millis(1),
            max_ttl: Duration::from_secs(3600),
        });
        for i in 0..5u32 {
            cache
                .put(format!("k{}", i), format!("v{}", i))
                .expect("should succeed");
        }
        thread::sleep(Duration::from_millis(100));
        let evicted = cache.evict_expired().expect("should succeed");
        assert_eq!(evicted, 5);
        assert_eq!(cache.len().expect("should succeed"), 0);
    }

    #[test]
    fn test_stats_hit_rate() {
        let cache = small_cache(10, 3600);
        cache
            .put("x".to_string(), "1".to_string())
            .expect("should succeed");
        let _ = cache.get(&"x".to_string()).expect("should succeed"); // hit
        let _ = cache.get(&"y".to_string()).expect("should succeed"); // miss

        let stats = cache.stats().expect("should succeed");
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_put_with_explicit_ttl() {
        let cache = QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(10).expect("should succeed"),
            default_ttl: Duration::from_secs(3600),
            min_ttl: Duration::from_millis(1),
            max_ttl: Duration::from_secs(86_400),
        });
        // Use a very short explicit TTL.
        cache
            .put_with_ttl("k".to_string(), "v".to_string(), Duration::from_millis(50))
            .expect("should succeed");
        assert!(cache
            .get(&"k".to_string())
            .expect("should succeed")
            .is_some());
        thread::sleep(Duration::from_millis(100));
        assert_eq!(cache.get(&"k".to_string()).expect("should succeed"), None);
    }

    #[test]
    fn test_clear() {
        let cache = small_cache(10, 3600);
        cache
            .put("a".to_string(), "1".to_string())
            .expect("should succeed");
        cache
            .put("b".to_string(), "2".to_string())
            .expect("should succeed");
        cache.clear().expect("should succeed");
        assert_eq!(cache.len().expect("should succeed"), 0);
    }

    #[test]
    fn test_thread_safe_concurrent_access() {
        let cache: QueryCache<String, usize> = QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(256).expect("should succeed"),
            default_ttl: Duration::from_secs(60),
            min_ttl: Duration::from_millis(1),
            max_ttl: Duration::from_secs(3600),
        });

        let handles: Vec<_> = (0..8_usize)
            .map(|t| {
                let c = cache.clone();
                thread::spawn(move || {
                    for i in 0..32_usize {
                        let key = format!("t{}k{}", t, i);
                        c.put(key.clone(), t * 100 + i).expect("put failed");
                        let _ = c.get(&key).expect("get failed");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = cache.stats().expect("should succeed");
        // Each thread inserted 32 entries and read them back, so ≥256 hits.
        assert!(stats.hits >= 256, "expected hits ≥256, got {}", stats.hits);
    }

    #[test]
    fn test_peek_entry_metadata() {
        let cache = small_cache(10, 3600);
        cache
            .put("k".to_string(), "v".to_string())
            .expect("should succeed");
        let hit_count = cache
            .peek_entry(&"k".to_string(), |e| e.hit_count)
            .expect("should succeed");
        assert_eq!(hit_count, Some(0)); // No reads yet via `get`.
        let _ = cache.get(&"k".to_string()).expect("should succeed");
        let hit_count2 = cache
            .peek_entry(&"k".to_string(), |e| e.hit_count)
            .expect("should succeed");
        assert_eq!(hit_count2, Some(1));
    }

    #[test]
    fn test_ttl_clamping() {
        let cache = QueryCache::new(QueryCacheConfig {
            capacity: NonZeroUsize::new(10).expect("should succeed"),
            default_ttl: Duration::from_secs(60),
            min_ttl: Duration::from_secs(10),
            max_ttl: Duration::from_secs(120),
        });
        // Request TTL below minimum – should be clamped to min_ttl (10s).
        cache
            .put_with_ttl("k".to_string(), "v".to_string(), Duration::from_millis(1))
            .expect("should succeed");
        // Entry should still be alive (clamped to 10s).
        let result = cache
            .peek_entry(&"k".to_string(), |e| e.ttl)
            .expect("should succeed");
        assert_eq!(result, Some(Duration::from_secs(10)));
    }
}
