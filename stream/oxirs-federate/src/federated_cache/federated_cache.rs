//! Cache for federated query results.
//!
//! Each cached entry tracks which endpoints contributed data, enabling
//! fine-grained invalidation when a specific endpoint's data changes.
//!
//! # Design
//!
//! - Thread-safe via `Arc<Mutex<...>>`
//! - LRU eviction when `max_entries` is reached
//! - TTL-based staleness: entries older than `ttl` are not returned
//! - Per-endpoint invalidation: removing all entries that used a given endpoint
//! - Stale eviction sweep: removes all expired entries in one pass

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─── Cache Key ────────────────────────────────────────────────────────────────

/// A composite cache key that uniquely identifies a federated query result.
///
/// Two queries are considered the same cache entry only if both the query
/// text *and* the set of endpoints queried match.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FederatedCacheKey {
    /// FNV/siphash of the normalised SPARQL query string
    pub query_hash: u64,
    /// Hash of the sorted set of endpoint IDs that participated
    pub endpoint_set_hash: u64,
}

impl FederatedCacheKey {
    /// Compute a key from a query string and a list of endpoint IDs.
    ///
    /// The endpoint list is sorted before hashing so that ordering does not
    /// affect the key.
    pub fn from_parts(query: &str, mut endpoints: Vec<String>) -> Self {
        endpoints.sort();

        let query_hash = {
            let mut h = DefaultHasher::new();
            query.hash(&mut h);
            h.finish()
        };
        let endpoint_set_hash = {
            let mut h = DefaultHasher::new();
            endpoints.hash(&mut h);
            h.finish()
        };

        Self {
            query_hash,
            endpoint_set_hash,
        }
    }
}

// ─── Cache Entry ──────────────────────────────────────────────────────────────

/// A stored federated query result with metadata.
#[derive(Debug, Clone)]
pub struct FederatedCacheEntry {
    /// The cache key used to store this entry
    pub key: FederatedCacheKey,
    /// Result bindings: each row is a map from variable name to value string
    pub result_bindings: Vec<HashMap<String, String>>,
    /// Endpoint IDs that contributed data to this result
    pub contributing_endpoints: Vec<String>,
    /// When this entry was created
    pub created_at: Instant,
    /// Maximum time-to-live for this entry
    pub ttl: Duration,
    /// Number of times this entry has been returned as a cache hit
    pub hit_count: u64,
    /// Last time this entry was accessed (for LRU tracking)
    pub last_accessed: Instant,
}

impl FederatedCacheEntry {
    /// Create a new cache entry.
    pub fn new(
        key: FederatedCacheKey,
        result_bindings: Vec<HashMap<String, String>>,
        contributing_endpoints: Vec<String>,
        ttl: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            key,
            result_bindings,
            contributing_endpoints,
            created_at: now,
            ttl,
            hit_count: 0,
            last_accessed: now,
        }
    }

    /// Return `true` if this entry has exceeded its TTL.
    pub fn is_stale(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

// ─── Cache Statistics ────────────────────────────────────────────────────────

/// Aggregate statistics for a [`FederatedQueryCache`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FederatedCacheStats {
    /// Total number of `get` calls
    pub total_requests: u64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses (entry absent or stale)
    pub misses: u64,
    /// Number of entries evicted via LRU
    pub lru_evictions: u64,
    /// Number of entries evicted via endpoint invalidation
    pub endpoint_invalidations: u64,
    /// Number of stale entries removed during sweep
    pub stale_evictions: u64,
}

impl FederatedCacheStats {
    /// Cache hit rate in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }
}

// ─── FederatedQueryCache ──────────────────────────────────────────────────────

/// Thread-safe cache for federated SPARQL query results.
///
/// # Features
///
/// - **TTL expiry**: entries are considered stale after `default_ttl`
/// - **LRU eviction**: when `max_entries` is exceeded the least-recently-used
///   entry is removed
/// - **Endpoint invalidation**: all entries that used a given endpoint can be
///   removed atomically, useful when an endpoint's data is known to have changed
/// - **Stale sweep**: removes all expired entries in a single pass
#[derive(Clone)]
pub struct FederatedQueryCache {
    inner: Arc<Mutex<CacheInner>>,
}

struct CacheInner {
    entries: HashMap<FederatedCacheKey, FederatedCacheEntry>,
    max_entries: usize,
    default_ttl: Duration,
    stats: FederatedCacheStats,
}

impl FederatedQueryCache {
    /// Create a new cache with the given capacity and default TTL.
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner {
                entries: HashMap::new(),
                max_entries,
                default_ttl: ttl,
                stats: FederatedCacheStats::default(),
            })),
        }
    }

    /// Retrieve cached result bindings for a key, if present and not stale.
    ///
    /// Returns `None` on a cache miss or if the entry has expired (and removes
    /// the stale entry as a side-effect).
    pub fn get(&self, key: &FederatedCacheKey) -> Option<Vec<HashMap<String, String>>> {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        inner.stats.total_requests += 1;

        // Check presence and staleness without holding a mutable borrow across stats updates.
        let present = inner.entries.contains_key(key);
        if !present {
            inner.stats.misses += 1;
            return None;
        }

        let stale = inner.entries.get(key).map(|e| e.is_stale()).unwrap_or(true);
        if stale {
            inner.stats.misses += 1;
            inner.entries.remove(key);
            return None;
        }

        // Entry is present and fresh - update access metadata then clone result.
        inner.stats.hits += 1;
        let now = Instant::now();
        let result = inner.entries.get_mut(key).map(|entry| {
            entry.hit_count += 1;
            entry.last_accessed = now;
            entry.result_bindings.clone()
        });
        result
    }

    /// Store result bindings for a key, using the default TTL.
    ///
    /// If the cache is full the least-recently-used entry is evicted first.
    pub fn put(
        &self,
        key: FederatedCacheKey,
        results: Vec<HashMap<String, String>>,
        endpoints: Vec<String>,
    ) {
        self.put_with_ttl(key, results, endpoints, None);
    }

    /// Store result bindings with an explicit TTL override.
    pub fn put_with_ttl(
        &self,
        key: FederatedCacheKey,
        results: Vec<HashMap<String, String>>,
        endpoints: Vec<String>,
        ttl_override: Option<Duration>,
    ) {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        let ttl = ttl_override.unwrap_or(inner.default_ttl);

        // Evict LRU entry if at capacity and the key is not already present
        if !inner.entries.contains_key(&key) && inner.entries.len() >= inner.max_entries {
            Self::evict_lru(&mut inner);
        }

        inner.entries.insert(
            key.clone(),
            FederatedCacheEntry::new(key, results, endpoints, ttl),
        );
    }

    /// Invalidate all cache entries that used `endpoint_id`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_endpoint(&self, endpoint_id: &str) -> usize {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        let before = inner.entries.len();
        inner.entries.retain(|_, entry| {
            !entry
                .contributing_endpoints
                .iter()
                .any(|e| e == endpoint_id)
        });
        let removed = before - inner.entries.len();
        inner.stats.endpoint_invalidations += removed as u64;
        removed
    }

    /// Return the current number of entries in the cache.
    pub fn size(&self) -> usize {
        self.inner
            .lock()
            .expect("cache lock poisoned")
            .entries
            .len()
    }

    /// Remove all entries that have exceeded their TTL.
    ///
    /// Returns the number of entries removed.
    pub fn evict_stale(&self) -> usize {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        let before = inner.entries.len();
        inner.entries.retain(|_, entry| !entry.is_stale());
        let removed = before - inner.entries.len();
        inner.stats.stale_evictions += removed as u64;
        removed
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> FederatedCacheStats {
        self.inner
            .lock()
            .expect("cache lock poisoned")
            .stats
            .clone()
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        inner.entries.clear();
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Evict the least-recently-used entry.
    fn evict_lru(inner: &mut CacheInner) {
        // Find the key with the oldest `last_accessed` timestamp
        let lru_key = inner
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            inner.entries.remove(&key);
            inner.stats.lru_evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_bindings(n: usize) -> Vec<HashMap<String, String>> {
        (0..n)
            .map(|i| {
                let mut row = HashMap::new();
                row.insert("x".to_string(), format!("value_{i}"));
                row
            })
            .collect()
    }

    fn make_key(query: &str, endpoints: &[&str]) -> FederatedCacheKey {
        FederatedCacheKey::from_parts(query, endpoints.iter().map(|s| s.to_string()).collect())
    }

    // ── FederatedCacheKey ────────────────────────────────────────────────────

    #[test]
    fn test_key_same_query_same_endpoints() {
        let k1 = make_key("SELECT * WHERE { ?s ?p ?o }", &["ep1", "ep2"]);
        let k2 = make_key("SELECT * WHERE { ?s ?p ?o }", &["ep2", "ep1"]); // different order
        assert_eq!(k1, k2, "keys should be order-independent");
    }

    #[test]
    fn test_key_different_queries() {
        let k1 = make_key("SELECT ?s WHERE { ?s a ex:A }", &["ep1"]);
        let k2 = make_key("SELECT ?s WHERE { ?s a ex:B }", &["ep1"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_key_different_endpoints() {
        let k1 = make_key("SELECT * WHERE { ?s ?p ?o }", &["ep1"]);
        let k2 = make_key("SELECT * WHERE { ?s ?p ?o }", &["ep2"]);
        assert_ne!(k1, k2);
    }

    // ── FederatedCacheEntry ──────────────────────────────────────────────────

    #[test]
    fn test_entry_not_stale_immediately() {
        let key = make_key("q", &["ep1"]);
        let entry = FederatedCacheEntry::new(
            key,
            make_bindings(2),
            vec!["ep1".to_string()],
            Duration::from_secs(60),
        );
        assert!(!entry.is_stale());
    }

    #[test]
    fn test_entry_stale_after_ttl() {
        let key = make_key("q", &["ep1"]);
        let mut entry = FederatedCacheEntry::new(
            key,
            make_bindings(2),
            vec!["ep1".to_string()],
            Duration::from_millis(1),
        );
        // Back-date the creation time
        entry.created_at = Instant::now() - Duration::from_secs(10);
        assert!(entry.is_stale());
    }

    // ── FederatedQueryCache ──────────────────────────────────────────────────

    #[test]
    fn test_cache_miss_on_empty() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));
        let key = make_key("q", &["ep1"]);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_hit_after_put() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));
        let key = make_key("SELECT * WHERE { ?s ?p ?o }", &["ep1", "ep2"]);
        let bindings = make_bindings(3);

        cache.put(
            key.clone(),
            bindings.clone(),
            vec!["ep1".to_string(), "ep2".to_string()],
        );
        let result = cache.get(&key);

        assert!(result.is_some());
        assert_eq!(result.expect("cache entry should be present").len(), 3);
    }

    #[test]
    fn test_cache_miss_after_stale() {
        let cache = FederatedQueryCache::new(100, Duration::from_millis(1));
        let key = make_key("q", &["ep1"]);
        cache.put(key.clone(), make_bindings(1), vec!["ep1".to_string()]);

        // Let the TTL expire
        thread::sleep(Duration::from_millis(10));

        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_invalidate_endpoint_removes_entries() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));

        let k1 = make_key("q1", &["ep1", "ep2"]);
        let k2 = make_key("q2", &["ep2", "ep3"]);
        let k3 = make_key("q3", &["ep3"]);

        cache.put(
            k1.clone(),
            make_bindings(1),
            vec!["ep1".to_string(), "ep2".to_string()],
        );
        cache.put(
            k2.clone(),
            make_bindings(1),
            vec!["ep2".to_string(), "ep3".to_string()],
        );
        cache.put(k3.clone(), make_bindings(1), vec!["ep3".to_string()]);

        let removed = cache.invalidate_endpoint("ep2");
        assert_eq!(removed, 2, "ep2 appears in k1 and k2");
        assert_eq!(cache.size(), 1, "only k3 remains");
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_invalidate_missing_endpoint_removes_nothing() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));
        let k = make_key("q", &["ep1"]);
        cache.put(k, make_bindings(1), vec!["ep1".to_string()]);

        let removed = cache.invalidate_endpoint("ep_nonexistent");
        assert_eq!(removed, 0);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_evict_stale() {
        let cache = FederatedQueryCache::new(100, Duration::from_millis(1));
        cache.put(
            make_key("q1", &["ep1"]),
            make_bindings(1),
            vec!["ep1".to_string()],
        );
        cache.put(
            make_key("q2", &["ep1"]),
            make_bindings(1),
            vec!["ep1".to_string()],
        );
        // Let both entries expire
        thread::sleep(Duration::from_millis(10));

        let removed = cache.evict_stale();
        assert_eq!(removed, 2);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_lru_eviction_on_capacity() {
        let cache = FederatedQueryCache::new(2, Duration::from_secs(60));

        let k1 = make_key("q1", &["ep1"]);
        let k2 = make_key("q2", &["ep1"]);
        let k3 = make_key("q3", &["ep1"]);

        cache.put(k1.clone(), make_bindings(1), vec!["ep1".to_string()]);
        cache.put(k2.clone(), make_bindings(1), vec!["ep1".to_string()]);

        // Access k2 to make k1 the LRU
        cache.get(&k2);

        // Inserting k3 should evict k1 (least recently used)
        cache.put(k3.clone(), make_bindings(1), vec!["ep1".to_string()]);

        assert_eq!(cache.size(), 2);
        // k1 was evicted
        assert!(cache.get(&k1).is_none());
        // k2 and k3 survive
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));
        let key = make_key("q", &["ep1"]);
        cache.put(key.clone(), make_bindings(1), vec!["ep1".to_string()]);

        cache.get(&key); // hit
        cache.get(&make_key("q2", &["ep1"])); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_empties_cache() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(60));
        cache.put(
            make_key("q", &["ep1"]),
            make_bindings(1),
            vec!["ep1".to_string()],
        );
        assert_eq!(cache.size(), 1);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_put_with_ttl_override() {
        let cache = FederatedQueryCache::new(100, Duration::from_secs(3600));
        let key = make_key("q", &["ep1"]);
        // Override TTL with 1ms
        cache.put_with_ttl(
            key.clone(),
            make_bindings(1),
            vec!["ep1".to_string()],
            Some(Duration::from_millis(1)),
        );
        thread::sleep(Duration::from_millis(10));
        assert!(cache.get(&key).is_none(), "entry should have expired");
    }

    #[test]
    fn test_cache_is_send_sync() {
        // Compile-time check: FederatedQueryCache must be Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FederatedQueryCache>();
    }
}
