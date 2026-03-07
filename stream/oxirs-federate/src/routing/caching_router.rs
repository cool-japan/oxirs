//! Caching router: caches routing decisions with TTL-based expiry.
//!
//! The [`CachingRouter`] wraps an inner routing step and memoises results,
//! keyed by the raw query string.  Entries expire after a configurable
//! time-to-live.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::semantic_router::Endpoint;

// ─── CacheEntry ───────────────────────────────────────────────────────────────

/// A cached routing decision.
#[derive(Debug, Clone)]
struct CacheEntry {
    endpoints: Vec<Endpoint>,
    created_at: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn new(endpoints: Vec<Endpoint>, ttl: Duration) -> Self {
        Self {
            endpoints,
            created_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

// ─── CachingRouterConfig ──────────────────────────────────────────────────────

/// Configuration for the caching router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingRouterConfig {
    /// Default TTL for cache entries.
    pub default_ttl_ms: u64,
    /// Maximum number of entries in the cache (0 = unlimited).
    pub max_entries: usize,
}

impl Default for CachingRouterConfig {
    fn default() -> Self {
        Self {
            default_ttl_ms: 60_000, // 1 minute
            max_entries: 1_024,
        }
    }
}

// ─── CacheStats ───────────────────────────────────────────────────────────────

/// Statistics for the caching router.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache look-up calls.
    pub total_lookups: u64,
    /// Number of cache hits (non-expired entry found).
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of invalidations (explicit or expiry-based evictions).
    pub invalidations: u64,
}

impl CacheStats {
    /// Hit rate in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

// ─── CachingRouter ────────────────────────────────────────────────────────────

/// A routing decision cache with TTL-based expiry.
///
/// The cache is keyed by the **full query string**; identical queries get
/// identical routing results until expiry.  Thread-safety is provided by the
/// caller via `Arc<Mutex<CachingRouter>>` if needed.
#[derive(Debug)]
pub struct CachingRouter {
    entries: HashMap<String, CacheEntry>,
    config: CachingRouterConfig,
    stats: CacheStats,
}

impl CachingRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: CachingRouterConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CachingRouterConfig::default())
    }

    /// Return cached routing endpoints for `query`, if still valid.
    ///
    /// Expired entries are removed and counted as misses.
    pub fn cached_route(&mut self, query: &str) -> Option<Vec<Endpoint>> {
        self.stats.total_lookups += 1;

        match self.entries.get(query) {
            Some(entry) if !entry.is_expired() => {
                self.stats.hits += 1;
                Some(entry.endpoints.clone())
            }
            Some(_expired) => {
                // Lazy eviction
                self.entries.remove(query);
                self.stats.invalidations += 1;
                self.stats.misses += 1;
                None
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Store routing endpoints for `query` with the default TTL.
    ///
    /// If the cache is at capacity (and `max_entries > 0`), the entry is NOT
    /// stored (capacity policy: reject rather than evict, for simplicity).
    pub fn cache_route(&mut self, query: impl Into<String>, endpoints: Vec<Endpoint>) {
        let ttl = Duration::from_millis(self.config.default_ttl_ms);
        self.cache_route_with_ttl(query, endpoints, ttl);
    }

    /// Store routing endpoints for `query` with a custom TTL.
    pub fn cache_route_with_ttl(
        &mut self,
        query: impl Into<String>,
        endpoints: Vec<Endpoint>,
        ttl: Duration,
    ) {
        let key = query.into();
        if self.config.max_entries > 0 && self.entries.len() >= self.config.max_entries {
            // At capacity: silently drop (could also evict-LRU, but keep simple)
            return;
        }
        self.entries.insert(key, CacheEntry::new(endpoints, ttl));
    }

    /// Explicitly invalidate the cache entry for `query`.
    pub fn invalidate(&mut self, query: &str) {
        if self.entries.remove(query).is_some() {
            self.stats.invalidations += 1;
        }
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_expired(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| !e.is_expired());
        let evicted = before - self.entries.len();
        self.stats.invalidations += evicted as u64;
        evicted
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        let count = self.entries.len() as u64;
        self.entries.clear();
        self.stats.invalidations += count;
    }

    /// Number of active (possibly expired) cache entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of non-expired cache entries.
    pub fn active_entry_count(&self) -> usize {
        self.entries.values().filter(|e| !e.is_expired()).count()
    }

    /// Current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// Configuration accessor.
    pub fn config(&self) -> &CachingRouterConfig {
        &self.config
    }

    /// Whether a non-expired entry exists for `query`.
    pub fn contains(&self, query: &str) -> bool {
        self.entries
            .get(query)
            .map(|e| !e.is_expired())
            .unwrap_or(false)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(id: &str) -> Endpoint {
        Endpoint::new(id, format!("http://{id}/sparql"))
    }

    fn long_ttl_config() -> CachingRouterConfig {
        CachingRouterConfig {
            default_ttl_ms: 60_000,
            max_entries: 100,
        }
    }

    fn short_ttl_config() -> CachingRouterConfig {
        CachingRouterConfig {
            default_ttl_ms: 1, // 1 ms — expires almost immediately
            max_entries: 100,
        }
    }

    #[test]
    fn test_cache_hit() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("SELECT *", vec![ep("ep1")]);
        let result = router.cached_route("SELECT *");
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut router = CachingRouter::new(long_ttl_config());
        let result = router.cached_route("SELECT *");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_ttl_expiry() {
        let mut router = CachingRouter::new(short_ttl_config());
        router.cache_route("SELECT *", vec![ep("ep1")]);
        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(10));
        let result = router.cached_route("SELECT *");
        assert!(result.is_none(), "expired entry should be a miss");
    }

    #[test]
    fn test_cache_invalidate() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.invalidate("q1");
        assert!(router.cached_route("q1").is_none());
    }

    #[test]
    fn test_cache_invalidate_nonexistent_no_panic() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.invalidate("nonexistent"); // should not panic
    }

    #[test]
    fn test_cache_clear() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.cache_route("q2", vec![ep("ep2")]);
        router.clear();
        assert_eq!(router.entry_count(), 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.cached_route("q1"); // hit
        router.cached_route("q2"); // miss
        let stats = router.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_zero_lookups_hit_rate() {
        let router = CachingRouter::new(long_ttl_config());
        assert_eq!(router.stats().hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_capacity_limit() {
        let config = CachingRouterConfig {
            default_ttl_ms: 60_000,
            max_entries: 2,
        };
        let mut router = CachingRouter::new(config);
        router.cache_route("q1", vec![ep("ep1")]);
        router.cache_route("q2", vec![ep("ep2")]);
        router.cache_route("q3", vec![ep("ep3")]); // should be silently dropped
        assert_eq!(router.entry_count(), 2);
    }

    #[test]
    fn test_cache_evict_expired() {
        let mut router = CachingRouter::new(short_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.cache_route("q2", vec![ep("ep2")]);
        std::thread::sleep(Duration::from_millis(10));
        let evicted = router.evict_expired();
        assert_eq!(evicted, 2);
        assert_eq!(router.entry_count(), 0);
    }

    #[test]
    fn test_cache_contains() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        assert!(router.contains("q1"));
        assert!(!router.contains("q2"));
    }

    #[test]
    fn test_cache_active_entry_count_excludes_expired() {
        let mut router = CachingRouter::default_config();
        router.cache_route_with_ttl("fresh", vec![ep("ep1")], Duration::from_secs(60));
        router.cache_route_with_ttl("stale", vec![ep("ep2")], Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(router.active_entry_count(), 1);
    }

    #[test]
    fn test_cache_reset_stats() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cached_route("q1"); // miss
        router.reset_stats();
        assert_eq!(router.stats().total_lookups, 0);
        assert_eq!(router.stats().misses, 0);
    }

    #[test]
    fn test_cache_route_with_ttl_custom() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route_with_ttl("q_custom", vec![ep("ep1")], Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        let result = router.cached_route("q_custom");
        assert!(result.is_none(), "custom short TTL should expire");
    }

    #[test]
    fn test_cache_config_accessor() {
        let config = long_ttl_config();
        let router = CachingRouter::new(config.clone());
        assert_eq!(router.config().default_ttl_ms, config.default_ttl_ms);
    }

    #[test]
    fn test_cache_multiple_endpoints() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("a"), ep("b"), ep("c")]);
        let result = router.cached_route("q1").expect("should hit");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cache_different_queries_independent() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("SELECT * WHERE { ?s ?p ?o }", vec![ep("ep1")]);
        router.cache_route("SELECT ?name WHERE { ?s foaf:name ?name }", vec![ep("ep2")]);
        let r1 = router
            .cached_route("SELECT * WHERE { ?s ?p ?o }")
            .expect("hit");
        let r2 = router
            .cached_route("SELECT ?name WHERE { ?s foaf:name ?name }")
            .expect("hit");
        assert_eq!(r1[0].id, "ep1");
        assert_eq!(r2[0].id, "ep2");
    }

    #[test]
    fn test_cache_stats_invalidations_on_clear() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.cache_route("q2", vec![ep("ep2")]);
        router.clear();
        assert_eq!(router.stats().invalidations, 2);
    }

    #[test]
    fn test_cache_stats_invalidation_on_explicit_invalidate() {
        let mut router = CachingRouter::new(long_ttl_config());
        router.cache_route("q1", vec![ep("ep1")]);
        router.invalidate("q1");
        assert_eq!(router.stats().invalidations, 1);
    }
}
