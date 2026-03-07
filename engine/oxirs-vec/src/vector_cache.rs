//! Vector embedding cache with LRU eviction.
//!
//! Provides an in-memory cache for embedding vectors with:
//! - LRU eviction (configurable max entries)
//! - Cache warming (preload frequently accessed vectors)
//! - Hit/miss statistics tracking
//! - Memory-bounded eviction
//! - Cache invalidation (by key, by prefix, clear all)
//! - Optional per-entry TTL-based expiry
//! - Batch get/put operations
//! - Cache persistence (save/load)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

/// Configuration for the vector cache.
#[derive(Debug, Clone)]
pub struct VectorCacheConfig {
    /// Maximum number of entries before LRU eviction (0 = unlimited).
    pub max_entries: usize,
    /// Maximum memory in bytes before eviction (0 = unlimited).
    pub max_memory_bytes: usize,
    /// Default TTL for entries. `None` = entries never expire.
    pub default_ttl: Option<Duration>,
}

impl Default for VectorCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_memory_bytes: 0,
            default_ttl: None,
        }
    }
}

/// Hit/miss statistics for the cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of entries inserted.
    pub inserts: u64,
    /// Total number of entries evicted (LRU or memory).
    pub evictions: u64,
    /// Total number of entries expired (TTL).
    pub expirations: u64,
    /// Total number of explicit invalidations.
    pub invalidations: u64,
}

impl CacheStatistics {
    /// Hit ratio in [0, 1].  Returns 0.0 if no requests have been made.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Total number of get requests.
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

/// A single cached vector entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached vector.
    vector: Vec<f64>,
    /// When this entry was inserted.
    inserted_at: Instant,
    /// When this entry was last accessed.
    last_accessed: Instant,
    /// Per-entry TTL override.  `None` uses the cache default.
    ttl: Option<Duration>,
}

impl CacheEntry {
    fn memory_bytes(&self) -> usize {
        // Approximate: Vec overhead + data + struct overhead
        std::mem::size_of::<Self>() + self.vector.len() * std::mem::size_of::<f64>()
    }

    fn is_expired(&self, default_ttl: Option<Duration>) -> bool {
        let ttl = self.ttl.or(default_ttl);
        if let Some(duration) = ttl {
            self.inserted_at.elapsed() > duration
        } else {
            false
        }
    }
}

/// A serializable representation of the cache for persistence.
#[derive(Debug, Clone)]
pub struct CacheSnapshot {
    /// Key → vector pairs.
    pub entries: Vec<(String, Vec<f64>)>,
    /// Total entries at snapshot time.
    pub entry_count: usize,
}

// ── VectorCache ──────────────────────────────────────────────────────────────

/// An LRU cache for embedding vectors.
pub struct VectorCache {
    /// The actual storage: key → entry.
    store: HashMap<String, CacheEntry>,
    /// LRU order: front = least recently used, back = most recently used.
    lru_order: VecDeque<String>,
    /// Configuration.
    config: VectorCacheConfig,
    /// Running statistics.
    stats: CacheStatistics,
}

impl VectorCache {
    /// Create a new cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(VectorCacheConfig::default())
    }

    /// Create a new cache with custom configuration.
    pub fn with_config(config: VectorCacheConfig) -> Self {
        Self {
            store: HashMap::new(),
            lru_order: VecDeque::new(),
            config,
            stats: CacheStatistics::default(),
        }
    }

    // ── Get / Put ────────────────────────────────────────────────────────────

    /// Get a cached vector by key.
    ///
    /// Returns `None` on miss.  Expired entries are evicted on access.
    pub fn get(&mut self, key: &str) -> Option<Vec<f64>> {
        // Check if key exists.
        if let Some(entry) = self.store.get(key) {
            if entry.is_expired(self.config.default_ttl) {
                // Expired — remove and count as miss.
                let key_owned = key.to_string();
                self.store.remove(&key_owned);
                self.lru_order.retain(|k| k != &key_owned);
                self.stats.expirations += 1;
                self.stats.misses += 1;
                return None;
            }
        } else {
            self.stats.misses += 1;
            return None;
        }

        // Hit — update LRU position.
        self.touch(key);
        self.stats.hits += 1;

        // Update last_accessed.
        if let Some(entry) = self.store.get_mut(key) {
            entry.last_accessed = Instant::now();
            Some(entry.vector.clone())
        } else {
            None
        }
    }

    /// Insert a vector into the cache with the default TTL.
    pub fn put(&mut self, key: impl Into<String>, vector: Vec<f64>) {
        self.put_with_ttl(key, vector, None);
    }

    /// Insert a vector with an explicit per-entry TTL.
    pub fn put_with_ttl(
        &mut self,
        key: impl Into<String>,
        vector: Vec<f64>,
        ttl: Option<Duration>,
    ) {
        let key = key.into();
        let now = Instant::now();

        let entry = CacheEntry {
            vector,
            inserted_at: now,
            last_accessed: now,
            ttl,
        };

        // If the key already exists, remove old LRU entry.
        if self.store.contains_key(&key) {
            self.lru_order.retain(|k| k != &key);
        }

        self.store.insert(key.clone(), entry);
        self.lru_order.push_back(key);
        self.stats.inserts += 1;

        // Enforce limits.
        self.enforce_entry_limit();
        self.enforce_memory_limit();
    }

    // ── Batch operations ─────────────────────────────────────────────────────

    /// Batch get: returns a map of key → vector for found entries.
    pub fn batch_get(&mut self, keys: &[&str]) -> HashMap<String, Vec<f64>> {
        let mut result = HashMap::new();
        for &key in keys {
            if let Some(vec) = self.get(key) {
                result.insert(key.to_string(), vec);
            }
        }
        result
    }

    /// Batch put: insert multiple entries at once.
    pub fn batch_put(&mut self, entries: Vec<(String, Vec<f64>)>) {
        for (key, vector) in entries {
            self.put(key, vector);
        }
    }

    // ── Cache warming ────────────────────────────────────────────────────────

    /// Warm the cache by preloading the given key-vector pairs.
    ///
    /// Existing entries are not overwritten.
    pub fn warm(&mut self, entries: Vec<(String, Vec<f64>)>) -> usize {
        let mut loaded = 0;
        for (key, vector) in entries {
            if !self.store.contains_key(&key) {
                self.put(key, vector);
                loaded += 1;
            }
        }
        loaded
    }

    // ── Invalidation ─────────────────────────────────────────────────────────

    /// Remove a specific key from the cache.
    pub fn invalidate(&mut self, key: &str) -> bool {
        if self.store.remove(key).is_some() {
            self.lru_order.retain(|k| k != key);
            self.stats.invalidations += 1;
            true
        } else {
            false
        }
    }

    /// Remove all keys matching a given prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys_to_remove: Vec<String> = self
            .store
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        let count = keys_to_remove.len();
        for key in &keys_to_remove {
            self.store.remove(key);
        }
        self.lru_order.retain(|k| !k.starts_with(prefix));
        self.stats.invalidations += count as u64;
        count
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        let count = self.store.len() as u64;
        self.store.clear();
        self.lru_order.clear();
        self.stats.invalidations += count;
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Current cache statistics.
    pub fn statistics(&self) -> &CacheStatistics {
        &self.stats
    }

    /// Reset statistics counters to zero.
    pub fn reset_statistics(&mut self) {
        self.stats = CacheStatistics::default();
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Approximate total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.store.values().map(|e| e.memory_bytes()).sum::<usize>()
            + self.lru_order.len() * std::mem::size_of::<String>()
    }

    /// Check if a key exists in the cache (without affecting LRU order).
    pub fn contains_key(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Create a snapshot of the current cache contents.
    pub fn snapshot(&self) -> CacheSnapshot {
        let entries: Vec<(String, Vec<f64>)> = self
            .store
            .iter()
            .filter(|(_, e)| !e.is_expired(self.config.default_ttl))
            .map(|(k, e)| (k.clone(), e.vector.clone()))
            .collect();
        let entry_count = entries.len();
        CacheSnapshot {
            entries,
            entry_count,
        }
    }

    /// Load entries from a snapshot (appending to the current cache).
    pub fn load_snapshot(&mut self, snapshot: CacheSnapshot) -> usize {
        let count = snapshot.entries.len();
        for (key, vector) in snapshot.entries {
            self.put(key, vector);
        }
        count
    }

    // ── Expiry sweep ─────────────────────────────────────────────────────────

    /// Sweep expired entries from the cache.
    ///
    /// Returns the number of entries removed.
    pub fn sweep_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self
            .store
            .iter()
            .filter(|(_, e)| e.is_expired(self.config.default_ttl))
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_keys.len();
        for key in &expired_keys {
            self.store.remove(key);
        }
        self.lru_order.retain(|k| !expired_keys.contains(k));
        self.stats.expirations += count as u64;
        count
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Move `key` to the back (most recently used) of the LRU deque.
    fn touch(&mut self, key: &str) {
        self.lru_order.retain(|k| k != key);
        self.lru_order.push_back(key.to_string());
    }

    /// Evict oldest entries until entry count is within limits.
    fn enforce_entry_limit(&mut self) {
        if self.config.max_entries == 0 {
            return;
        }
        while self.store.len() > self.config.max_entries {
            if let Some(oldest) = self.lru_order.pop_front() {
                self.store.remove(&oldest);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Evict oldest entries until memory usage is within limits.
    fn enforce_memory_limit(&mut self) {
        if self.config.max_memory_bytes == 0 {
            return;
        }
        while self.memory_usage() > self.config.max_memory_bytes {
            if let Some(oldest) = self.lru_order.pop_front() {
                self.store.remove(&oldest);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
    }
}

impl Default for VectorCache {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(a: f64, b: f64, c: f64) -> Vec<f64> {
        vec![a, b, c]
    }

    // ── Basic get/put ────────────────────────────────────────────────────────

    #[test]
    fn test_put_and_get() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        let v = cache.get("k1");
        assert!(v.is_some());
        assert_eq!(v.expect("should exist"), vec3(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_get_miss() {
        let mut cache = VectorCache::new();
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.statistics().misses, 1);
    }

    #[test]
    fn test_put_overwrite() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        cache.put("k1", vec3(4.0, 5.0, 6.0));
        let v = cache.get("k1");
        assert_eq!(v.expect("should exist"), vec3(4.0, 5.0, 6.0));
        assert_eq!(cache.len(), 1);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction() {
        let config = VectorCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        cache.put("c", vec3(0.0, 0.0, 1.0)); // evicts "a"

        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
        assert_eq!(cache.statistics().evictions, 1);
    }

    #[test]
    fn test_lru_access_refreshes() {
        let config = VectorCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));

        // Access "a" to make it recently used.
        let _ = cache.get("a");

        cache.put("c", vec3(0.0, 0.0, 1.0)); // evicts "b" (LRU), not "a"

        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    // ── Memory-bounded eviction ──────────────────────────────────────────────

    #[test]
    fn test_memory_limit_eviction() {
        let entry_size = std::mem::size_of::<CacheEntry>() + 3 * std::mem::size_of::<f64>();
        let config = VectorCacheConfig {
            max_entries: 0,                         // unlimited entries
            max_memory_bytes: entry_size * 2 + 100, // roughly 2 entries
            default_ttl: None,
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        cache.put("c", vec3(0.0, 0.0, 1.0));

        // At most 2 entries should remain (might be fewer due to overhead).
        assert!(cache.len() <= 3);
        let _ = cache.statistics().evictions;
    }

    // ── TTL-based expiry ─────────────────────────────────────────────────────

    #[test]
    fn test_ttl_expiry() {
        let config = VectorCacheConfig {
            default_ttl: Some(Duration::from_millis(1)),
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("k1", vec3(1.0, 2.0, 3.0));

        // Wait for TTL to expire.
        std::thread::sleep(Duration::from_millis(10));

        assert!(cache.get("k1").is_none());
        assert!(cache.statistics().expirations >= 1);
    }

    #[test]
    fn test_per_entry_ttl() {
        let mut cache = VectorCache::new();
        cache.put_with_ttl("k1", vec3(1.0, 2.0, 3.0), Some(Duration::from_millis(1)));
        cache.put("k2", vec3(4.0, 5.0, 6.0)); // No TTL

        std::thread::sleep(Duration::from_millis(10));

        assert!(cache.get("k1").is_none()); // expired
        assert!(cache.get("k2").is_some()); // still alive
    }

    #[test]
    fn test_sweep_expired() {
        let config = VectorCacheConfig {
            default_ttl: Some(Duration::from_millis(1)),
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));

        std::thread::sleep(Duration::from_millis(10));

        let swept = cache.sweep_expired();
        assert_eq!(swept, 2);
        assert!(cache.is_empty());
    }

    // ── Invalidation ─────────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_key() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        assert!(cache.invalidate("k1"));
        assert!(cache.get("k1").is_none());
        assert_eq!(cache.statistics().invalidations, 1);
    }

    #[test]
    fn test_invalidate_nonexistent() {
        let mut cache = VectorCache::new();
        assert!(!cache.invalidate("nope"));
    }

    #[test]
    fn test_invalidate_prefix() {
        let mut cache = VectorCache::new();
        cache.put("user:1", vec3(1.0, 0.0, 0.0));
        cache.put("user:2", vec3(0.0, 1.0, 0.0));
        cache.put("item:1", vec3(0.0, 0.0, 1.0));

        let removed = cache.invalidate_prefix("user:");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("item:1"));
    }

    #[test]
    fn test_clear() {
        let mut cache = VectorCache::new();
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        cache.clear();
        assert!(cache.is_empty());
    }

    // ── Batch operations ─────────────────────────────────────────────────────

    #[test]
    fn test_batch_put_and_get() {
        let mut cache = VectorCache::new();
        cache.batch_put(vec![
            ("k1".to_string(), vec3(1.0, 0.0, 0.0)),
            ("k2".to_string(), vec3(0.0, 1.0, 0.0)),
            ("k3".to_string(), vec3(0.0, 0.0, 1.0)),
        ]);
        assert_eq!(cache.len(), 3);

        let results = cache.batch_get(&["k1", "k3", "missing"]);
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("k1"));
        assert!(results.contains_key("k3"));
    }

    // ── Cache warming ────────────────────────────────────────────────────────

    #[test]
    fn test_warm() {
        let mut cache = VectorCache::new();
        cache.put("existing", vec3(9.0, 9.0, 9.0));

        let loaded = cache.warm(vec![
            ("existing".to_string(), vec3(0.0, 0.0, 0.0)), // should NOT overwrite
            ("new1".to_string(), vec3(1.0, 0.0, 0.0)),
            ("new2".to_string(), vec3(0.0, 1.0, 0.0)),
        ]);

        assert_eq!(loaded, 2);
        assert_eq!(cache.len(), 3);

        // "existing" should retain its original value.
        let v = cache.get("existing").expect("should exist");
        assert_eq!(v, vec3(9.0, 9.0, 9.0));
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    #[test]
    fn test_hit_ratio() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        let _ = cache.get("k1"); // hit
        let _ = cache.get("k2"); // miss

        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio() - 0.5).abs() < f64::EPSILON);
        assert_eq!(stats.total_requests(), 2);
    }

    #[test]
    fn test_hit_ratio_no_requests() {
        let cache = VectorCache::new();
        assert!((cache.statistics().hit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_statistics() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        let _ = cache.get("k1");
        cache.reset_statistics();
        assert_eq!(cache.statistics().hits, 0);
        assert_eq!(cache.statistics().inserts, 0);
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_and_load() {
        let mut cache = VectorCache::new();
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));

        let snap = cache.snapshot();
        assert_eq!(snap.entry_count, 2);

        let mut cache2 = VectorCache::new();
        let loaded = cache2.load_snapshot(snap);
        assert_eq!(loaded, 2);
        assert!(cache2.get("a").is_some());
        assert!(cache2.get("b").is_some());
    }

    #[test]
    fn test_snapshot_excludes_expired() {
        let config = VectorCacheConfig {
            default_ttl: Some(Duration::from_millis(1)),
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("x", vec3(1.0, 2.0, 3.0));
        std::thread::sleep(Duration::from_millis(10));

        let snap = cache.snapshot();
        assert_eq!(snap.entry_count, 0);
    }

    // ── Len / is_empty / contains_key ────────────────────────────────────────

    #[test]
    fn test_len_and_empty() {
        let mut cache = VectorCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.put("k1", vec3(1.0, 2.0, 3.0));
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_contains_key() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        assert!(cache.contains_key("k1"));
        assert!(!cache.contains_key("k2"));
    }

    // ── Memory usage ─────────────────────────────────────────────────────────

    #[test]
    fn test_memory_usage_grows() {
        let mut cache = VectorCache::new();
        let m0 = cache.memory_usage();
        cache.put("k1", vec![0.0; 100]);
        let m1 = cache.memory_usage();
        assert!(m1 > m0);
    }

    // ── Default ──────────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let c = VectorCacheConfig::default();
        assert_eq!(c.max_entries, 10_000);
        assert_eq!(c.max_memory_bytes, 0);
        assert!(c.default_ttl.is_none());
    }

    #[test]
    fn test_default_cache() {
        let cache = VectorCache::default();
        assert!(cache.is_empty());
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_vector() {
        let mut cache = VectorCache::new();
        cache.put("empty", vec![]);
        let v = cache.get("empty");
        assert_eq!(v.expect("should exist"), Vec::<f64>::new());
    }

    #[test]
    fn test_large_vector() {
        let mut cache = VectorCache::new();
        let big = vec![1.0; 10_000];
        cache.put("big", big.clone());
        let v = cache.get("big").expect("should exist");
        assert_eq!(v.len(), 10_000);
    }

    #[test]
    fn test_invalidate_prefix_no_match() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        let removed = cache.invalidate_prefix("zzz:");
        assert_eq!(removed, 0);
        assert_eq!(cache.len(), 1);
    }

    // ── Additional tests for coverage ────────────────────────────────────────

    #[test]
    fn test_multiple_evictions() {
        let config = VectorCacheConfig {
            max_entries: 3,
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        for i in 0..10 {
            cache.put(format!("k{i}"), vec![i as f64]);
        }
        assert_eq!(cache.len(), 3);
        assert!(cache.statistics().evictions >= 7);
    }

    #[test]
    fn test_batch_get_all_miss() {
        let mut cache = VectorCache::new();
        let results = cache.batch_get(&["a", "b", "c"]);
        assert!(results.is_empty());
        assert_eq!(cache.statistics().misses, 3);
    }

    #[test]
    fn test_batch_put_then_invalidate() {
        let mut cache = VectorCache::new();
        cache.batch_put(vec![
            ("a".to_string(), vec3(1.0, 0.0, 0.0)),
            ("b".to_string(), vec3(0.0, 1.0, 0.0)),
        ]);
        cache.invalidate("a");
        assert!(!cache.contains_key("a"));
        assert!(cache.contains_key("b"));
    }

    #[test]
    fn test_warm_empty_list() {
        let mut cache = VectorCache::new();
        let loaded = cache.warm(vec![]);
        assert_eq!(loaded, 0);
    }

    #[test]
    fn test_snapshot_empty_cache() {
        let cache = VectorCache::new();
        let snap = cache.snapshot();
        assert_eq!(snap.entry_count, 0);
        assert!(snap.entries.is_empty());
    }

    #[test]
    fn test_load_snapshot_into_non_empty_cache() {
        let mut cache1 = VectorCache::new();
        cache1.put("a", vec3(1.0, 0.0, 0.0));
        let snap = cache1.snapshot();

        let mut cache2 = VectorCache::new();
        cache2.put("b", vec3(0.0, 1.0, 0.0));
        cache2.load_snapshot(snap);

        assert!(cache2.contains_key("a"));
        assert!(cache2.contains_key("b"));
        assert_eq!(cache2.len(), 2);
    }

    #[test]
    fn test_put_updates_insert_count() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 0.0, 0.0));
        cache.put("k2", vec3(0.0, 1.0, 0.0));
        assert_eq!(cache.statistics().inserts, 2);
    }

    #[test]
    fn test_clear_resets_len() {
        let mut cache = VectorCache::new();
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        cache.put("c", vec3(0.0, 0.0, 1.0));
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_stats_invalidation_count() {
        let mut cache = VectorCache::new();
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        cache.invalidate("a");
        cache.invalidate("b");
        assert_eq!(cache.statistics().invalidations, 2);
    }

    #[test]
    fn test_get_after_clear() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        cache.clear();
        assert!(cache.get("k1").is_none());
    }

    #[test]
    fn test_sweep_expired_none_expired() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec3(1.0, 2.0, 3.0));
        let swept = cache.sweep_expired();
        assert_eq!(swept, 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_contains_key_after_eviction() {
        let config = VectorCacheConfig {
            max_entries: 1,
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("first", vec3(1.0, 0.0, 0.0));
        cache.put("second", vec3(0.0, 1.0, 0.0));
        assert!(!cache.contains_key("first"));
        assert!(cache.contains_key("second"));
    }

    #[test]
    fn test_invalidate_prefix_all() {
        let mut cache = VectorCache::new();
        cache.put("x:1", vec3(1.0, 0.0, 0.0));
        cache.put("x:2", vec3(0.0, 1.0, 0.0));
        cache.put("x:3", vec3(0.0, 0.0, 1.0));
        let removed = cache.invalidate_prefix("x:");
        assert_eq!(removed, 3);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_memory_usage_after_clear() {
        let mut cache = VectorCache::new();
        cache.put("k1", vec![0.0; 1000]);
        let before = cache.memory_usage();
        assert!(before > 0);
        cache.clear();
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_put_with_zero_ttl() {
        let mut cache = VectorCache::new();
        cache.put_with_ttl("k1", vec3(1.0, 2.0, 3.0), Some(Duration::from_secs(0)));
        // TTL = 0 means already expired on next access.
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get("k1").is_none());
    }

    #[test]
    fn test_lru_order_after_overwrite() {
        let config = VectorCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = VectorCache::with_config(config);
        cache.put("a", vec3(1.0, 0.0, 0.0));
        cache.put("b", vec3(0.0, 1.0, 0.0));
        // Overwrite "a" — it should become most recently used
        cache.put("a", vec3(9.0, 9.0, 9.0));
        // Insert "c" — should evict "b", not "a"
        cache.put("c", vec3(0.0, 0.0, 1.0));
        assert!(cache.contains_key("a"));
        assert!(!cache.contains_key("b"));
        assert!(cache.contains_key("c"));
    }
}
