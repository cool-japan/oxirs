//! # Per-endpoint TTL cache for federated sub-results
//!
//! Stores the results of stable federated SERVICE invocations keyed by
//! `(endpoint_url, sparql_text)` with a configurable time-to-live.  Designed
//! to be hot-path safe: lookup is `O(1)` plus a single timestamp comparison.
//!
//! ## Design notes
//!
//! - **In-memory only.**  Persistence is deliberately out of scope; restart
//!   wipes the cache and the system warms up again as queries flow.
//! - **No compression.**  Per the COOLJAPAN Pure Rust Policy, compression
//!   would have to use OxiARC (no zstd/zip/flate2/brotli/snap).  Until
//!   payloads grow large enough to justify the overhead, plain in-memory
//!   storage wins on latency and simplicity.
//! - **Bounded size.**  An LRU-style purge runs whenever the cache exceeds
//!   `max_entries` after an insert.  Eviction takes the entry with the
//!   oldest `inserted_at` timestamp.
//! - **TTL pruning.**  `prune_expired` is idempotent and cheap; call it on a
//!   timer or as part of a maintenance loop.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// A composite cache key.  Two queries against the same endpoint with
/// identical SPARQL text are considered equivalent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointCacheKey {
    /// Federated endpoint URL.
    pub endpoint: String,
    /// SPARQL text exactly as sent over the wire.
    pub sparql: String,
}

impl EndpointCacheKey {
    /// Build a key from any string-like inputs.
    pub fn new(endpoint: impl Into<String>, sparql: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            sparql: sparql.into(),
        }
    }
}

/// Cached payload + timing metadata.
#[derive(Debug, Clone)]
pub struct CachedSubresult<V> {
    /// The actual cached value.
    pub value: V,
    /// Wall-clock instant of insertion (used for TTL).
    pub inserted_at: Instant,
    /// Number of successful hits this entry has served.
    pub hits: u64,
    /// Estimated size in bytes (informational; eviction policy uses age).
    pub byte_size: usize,
}

impl<V> CachedSubresult<V> {
    fn new(value: V, byte_size: usize) -> Self {
        Self {
            value,
            inserted_at: Instant::now(),
            hits: 0,
            byte_size,
        }
    }

    /// Whether the entry has exceeded the supplied TTL.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        Instant::now().duration_since(self.inserted_at) > ttl
    }

    /// Borrow the wrapped value.
    pub fn value(&self) -> &V {
        &self.value
    }
}

/// Configuration for an [`EndpointCache`].
#[derive(Debug, Clone)]
pub struct EndpointCacheConfig {
    /// Time-to-live for entries.
    pub ttl: Duration,
    /// Maximum number of entries before LRU eviction kicks in.
    pub max_entries: usize,
    /// Per-endpoint TTL overrides.  Used when a particular service is known
    /// to update slowly (long TTL) or rapidly (short TTL).
    pub per_endpoint_ttl: HashMap<String, Duration>,
}

impl Default for EndpointCacheConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(300), // 5 minutes
            max_entries: 1024,
            per_endpoint_ttl: HashMap::new(),
        }
    }
}

impl EndpointCacheConfig {
    /// Create a config with a uniform TTL.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl,
            ..Self::default()
        }
    }

    /// Set a per-endpoint TTL override.
    pub fn override_ttl(mut self, endpoint: impl Into<String>, ttl: Duration) -> Self {
        self.per_endpoint_ttl.insert(endpoint.into(), ttl);
        self
    }

    /// Resolve the TTL applicable to a given endpoint URL.
    pub fn ttl_for(&self, endpoint: &str) -> Duration {
        self.per_endpoint_ttl
            .get(endpoint)
            .copied()
            .unwrap_or(self.ttl)
    }
}

/// Counters returned by [`EndpointCache::stats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointCacheStats {
    /// Total successful lookups.
    pub hits: u64,
    /// Total failed lookups (key not present or expired).
    pub misses: u64,
    /// Total inserts.
    pub inserts: u64,
    /// Total evictions due to LRU pressure.
    pub evictions: u64,
    /// Total expirations from `prune_expired`.
    pub expired: u64,
    /// Current entry count.
    pub size: usize,
}

impl EndpointCacheStats {
    /// Hit ratio in `[0.0, 1.0]`, or `0.0` if no lookups have been observed.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// Concurrent-safe cache of federated sub-results.
///
/// Generic over the cached value type `V`; in production we usually store an
/// owned `Vec<u8>` or a serialized `SparqlBindings` blob.
#[derive(Clone)]
pub struct EndpointCache<V: Clone> {
    inner: Arc<Mutex<EndpointCacheInner<V>>>,
    config: EndpointCacheConfig,
}

struct EndpointCacheInner<V: Clone> {
    entries: HashMap<EndpointCacheKey, CachedSubresult<V>>,
    stats: EndpointCacheStats,
}

impl<V: Clone> EndpointCache<V> {
    /// Build a new cache with the supplied config.
    pub fn new(config: EndpointCacheConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EndpointCacheInner {
                entries: HashMap::new(),
                stats: EndpointCacheStats::default(),
            })),
            config,
        }
    }

    /// Build a cache with the default config (5 min TTL, 1024 entries).
    pub fn with_defaults() -> Self {
        Self::new(EndpointCacheConfig::default())
    }

    /// Read the borrowed config.
    pub fn config(&self) -> &EndpointCacheConfig {
        &self.config
    }

    /// Replace the running config with a new one (e.g. to extend TTL after
    /// observing endpoint stability).
    pub fn set_config(&mut self, config: EndpointCacheConfig) {
        self.config = config;
    }

    /// Look up an entry.  Returns `None` on miss or if the entry has expired.
    /// Expired entries are *not* removed by `get`; call `prune_expired` for
    /// that.
    pub fn get(&self, key: &EndpointCacheKey) -> Option<V> {
        let ttl = self.config.ttl_for(&key.endpoint);
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let result = guard
            .entries
            .get_mut(key)
            .filter(|e| !e.is_expired(ttl))
            .map(|e| {
                e.hits += 1;
                e.value.clone()
            });
        if result.is_some() {
            guard.stats.hits += 1;
        } else {
            guard.stats.misses += 1;
        }
        result
    }

    /// Insert or replace an entry.  Triggers LRU eviction if over capacity.
    pub fn insert(&self, key: EndpointCacheKey, value: V, byte_size: usize) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard
            .entries
            .insert(key, CachedSubresult::new(value, byte_size));
        guard.stats.inserts += 1;
        guard.stats.size = guard.entries.len();
        // Capacity check.
        while guard.entries.len() > self.config.max_entries {
            if !evict_oldest(&mut guard) {
                break;
            }
        }
        guard.stats.size = guard.entries.len();
    }

    /// Remove an entry by key.  No-op if absent.
    pub fn invalidate(&self, key: &EndpointCacheKey) -> bool {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let removed = guard.entries.remove(key).is_some();
        if removed {
            guard.stats.size = guard.entries.len();
        }
        removed
    }

    /// Drop every cached entry whose endpoint matches `endpoint`.  Returns
    /// the number of removals.
    pub fn invalidate_endpoint(&self, endpoint: &str) -> usize {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let before = guard.entries.len();
        guard.entries.retain(|k, _| k.endpoint != endpoint);
        let removed = before - guard.entries.len();
        guard.stats.size = guard.entries.len();
        removed
    }

    /// Drop every entry that has exceeded its TTL.  Returns the number of
    /// removals.
    pub fn prune_expired(&self) -> usize {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let cfg = &self.config;
        let before = guard.entries.len();
        guard
            .entries
            .retain(|k, v| !v.is_expired(cfg.ttl_for(&k.endpoint)));
        let removed = before - guard.entries.len();
        guard.stats.expired += removed as u64;
        guard.stats.size = guard.entries.len();
        removed
    }

    /// Drop every entry, preserving stats.
    pub fn clear(&self) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.entries.clear();
        guard.stats.size = 0;
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> EndpointCacheStats {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.stats.clone()
    }

    /// Total number of entries currently held.
    pub fn len(&self) -> usize {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.entries.len()
    }

    /// Whether the cache is currently empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn evict_oldest<V: Clone>(guard: &mut std::sync::MutexGuard<'_, EndpointCacheInner<V>>) -> bool {
    // Find oldest entry by inserted_at.
    let oldest_key = guard
        .entries
        .iter()
        .min_by_key(|(_, v)| v.inserted_at)
        .map(|(k, _)| k.clone());
    if let Some(k) = oldest_key {
        guard.entries.remove(&k);
        guard.stats.evictions += 1;
        true
    } else {
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn key(ep: &str, q: &str) -> EndpointCacheKey {
        EndpointCacheKey::new(ep, q)
    }

    #[test]
    fn empty_get_returns_none() {
        let cache: EndpointCache<String> = EndpointCache::with_defaults();
        assert!(cache.get(&key("http://a.example", "SELECT 1")).is_none());
        assert!(cache.is_empty());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn insert_and_get_round_trip() {
        let cache: EndpointCache<String> = EndpointCache::with_defaults();
        let k = key("http://a.example", "SELECT 1");
        cache.insert(k.clone(), "hello".into(), 5);
        assert_eq!(cache.get(&k).as_deref(), Some("hello"));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn ttl_expires_entry() {
        let cfg = EndpointCacheConfig::with_ttl(Duration::from_millis(20));
        let cache: EndpointCache<u32> = EndpointCache::new(cfg);
        let k = key("http://a.example", "SELECT 1");
        cache.insert(k.clone(), 7, 1);
        thread::sleep(Duration::from_millis(40));
        assert!(cache.get(&k).is_none(), "entry should have expired");
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn prune_expired_removes_stale() {
        let cfg = EndpointCacheConfig::with_ttl(Duration::from_millis(20));
        let cache: EndpointCache<u32> = EndpointCache::new(cfg);
        cache.insert(key("a", "q1"), 1, 1);
        cache.insert(key("a", "q2"), 2, 1);
        thread::sleep(Duration::from_millis(40));
        let removed = cache.prune_expired();
        assert_eq!(removed, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_specific_key() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        let k = key("a", "q1");
        cache.insert(k.clone(), 1, 1);
        assert!(cache.invalidate(&k));
        assert!(cache.get(&k).is_none());
        assert!(!cache.invalidate(&k), "second invalidate is no-op");
    }

    #[test]
    fn invalidate_endpoint_removes_all_for_endpoint() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        cache.insert(key("a", "q1"), 1, 1);
        cache.insert(key("a", "q2"), 2, 1);
        cache.insert(key("b", "q1"), 3, 1);
        let n = cache.invalidate_endpoint("a");
        assert_eq!(n, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lru_eviction_when_over_capacity() {
        let cfg = EndpointCacheConfig {
            max_entries: 2,
            ..EndpointCacheConfig::default()
        };
        let cache: EndpointCache<u32> = EndpointCache::new(cfg);
        cache.insert(key("a", "q1"), 1, 1);
        thread::sleep(Duration::from_millis(2));
        cache.insert(key("a", "q2"), 2, 1);
        thread::sleep(Duration::from_millis(2));
        cache.insert(key("a", "q3"), 3, 1); // evicts q1
        assert!(cache.get(&key("a", "q1")).is_none());
        assert!(cache.get(&key("a", "q2")).is_some());
        assert!(cache.get(&key("a", "q3")).is_some());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn per_endpoint_ttl_override() {
        let cfg = EndpointCacheConfig::with_ttl(Duration::from_secs(60))
            .override_ttl("a", Duration::from_millis(20));
        let cache: EndpointCache<u32> = EndpointCache::new(cfg);
        cache.insert(key("a", "q1"), 1, 1);
        cache.insert(key("b", "q1"), 2, 1);
        thread::sleep(Duration::from_millis(40));
        // a has shorter TTL → expired.
        assert!(cache.get(&key("a", "q1")).is_none());
        // b has the default 60s TTL → still cached.
        assert!(cache.get(&key("b", "q1")).is_some());
    }

    #[test]
    fn stats_hit_ratio() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        let k = key("a", "q");
        cache.insert(k.clone(), 1, 1);
        let _ = cache.get(&k); // hit
        let _ = cache.get(&key("a", "missing")); // miss
        let stats = cache.stats();
        assert!((stats.hit_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn clear_resets_entries_only() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        cache.insert(key("a", "q"), 1, 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn len_tracks_entries() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        cache.insert(key("a", "q1"), 1, 1);
        cache.insert(key("a", "q2"), 2, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn duplicate_insert_overwrites_value() {
        let cache: EndpointCache<u32> = EndpointCache::with_defaults();
        let k = key("a", "q");
        cache.insert(k.clone(), 1, 1);
        cache.insert(k.clone(), 2, 1);
        assert_eq!(cache.get(&k), Some(2));
    }

    #[test]
    fn cache_stats_default_zero() {
        let s = EndpointCacheStats::default();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert!((s.hit_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn config_ttl_for_default() {
        let cfg = EndpointCacheConfig::default();
        assert_eq!(cfg.ttl_for("anywhere"), Duration::from_secs(300));
    }
}
