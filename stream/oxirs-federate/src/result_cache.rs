//! Federated query result caching with TTL and LRU eviction.
//!
//! `ResultCache` stores query results keyed by a 64-bit FNV-1a hash of the
//! (query, endpoint) pair. Entries expire after their TTL and are evicted LRU
//! when the cache is full.

use std::collections::{HashMap, VecDeque};

/// A cached federated query result.
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub query_hash: u64,
    pub endpoint: String,
    pub rows: Vec<HashMap<String, String>>,
    pub cached_at_ms: u64,
    pub ttl_ms: u64,
}

impl CachedResult {
    /// Create a new cached result.
    pub fn new(
        query_hash: u64,
        endpoint: impl Into<String>,
        rows: Vec<HashMap<String, String>>,
        cached_at_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            query_hash,
            endpoint: endpoint.into(),
            rows,
            cached_at_ms,
            ttl_ms,
        }
    }

    /// Whether this entry has expired.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.cached_at_ms) >= self.ttl_ms
    }
}

/// Accumulated cache access statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub size: usize,
}

/// TTL + LRU result cache for federated queries.
pub struct ResultCache {
    entries: HashMap<u64, CachedResult>,
    /// Front = most recently used; back = least recently used.
    lru_order: VecDeque<u64>,
    max_size: usize,
    stats: CacheStats,
}

impl ResultCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            max_size: max_size.max(1),
            stats: CacheStats::default(),
        }
    }

    /// Look up a result. Returns `Some(&CachedResult)` on hit (unexpired),
    /// `None` on miss or expiry. Updates hit/miss statistics.
    pub fn get(&mut self, query_hash: u64, now_ms: u64) -> Option<&CachedResult> {
        if let Some(entry) = self.entries.get(&query_hash) {
            if entry.is_expired(now_ms) {
                // treat expired as miss; leave removal to evict_expired
                self.stats.misses += 1;
                return None;
            }
            // Move to front of LRU
            self.lru_order.retain(|&h| h != query_hash);
            self.lru_order.push_front(query_hash);
            self.stats.hits += 1;
            self.stats.size = self.entries.len();
            return self.entries.get(&query_hash);
        }
        self.stats.misses += 1;
        None
    }

    /// Insert or replace a cached result, evicting LRU entries if needed.
    pub fn insert(&mut self, result: CachedResult) {
        let hash = result.query_hash;
        // If already present, remove from LRU order first
        if self.entries.contains_key(&hash) {
            self.lru_order.retain(|&h| h != hash);
        }
        // Evict LRU entries until there is room
        while self.entries.len() >= self.max_size {
            if let Some(lru_hash) = self.lru_order.pop_back() {
                self.entries.remove(&lru_hash);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
        self.lru_order.push_front(hash);
        self.entries.insert(hash, result);
        self.stats.size = self.entries.len();
    }

    /// Remove all entries whose TTL has expired. Returns the number evicted.
    pub fn evict_expired(&mut self, now_ms: u64) -> usize {
        let expired: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, v)| v.is_expired(now_ms))
            .map(|(k, _)| *k)
            .collect();
        let count = expired.len();
        for hash in &expired {
            self.entries.remove(hash);
            self.lru_order.retain(|&h| h != *hash);
            self.stats.evictions += 1;
        }
        self.stats.size = self.entries.len();
        count
    }

    /// Remove all entries for the given endpoint. Returns the count removed.
    pub fn invalidate_endpoint(&mut self, endpoint: &str) -> usize {
        let to_remove: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, v)| v.endpoint == endpoint)
            .map(|(k, _)| *k)
            .collect();
        let count = to_remove.len();
        for hash in &to_remove {
            self.entries.remove(hash);
            self.lru_order.retain(|&h| h != *hash);
            self.stats.evictions += count;
        }
        self.stats.size = self.entries.len();
        count
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats.size = 0;
    }

    /// Current cache statistics (snapshot).
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Hit rate in [0.0, 1.0]. Returns 0.0 if no lookups yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }

    /// Number of entries currently in the cache.
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Compute FNV-1a 64-bit hash for (query, endpoint).
    pub fn hash_query(query: &str, endpoint: &str) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in query.bytes().chain(b"|".iter().copied()).chain(endpoint.bytes()) {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

impl Default for ResultCache {
    fn default() -> Self {
        Self::new(256)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(hash: u64, endpoint: &str, cached_at: u64, ttl: u64) -> CachedResult {
        CachedResult::new(hash, endpoint, vec![], cached_at, ttl)
    }

    fn make_result_rows(
        hash: u64,
        endpoint: &str,
        rows: Vec<HashMap<String, String>>,
        cached_at: u64,
        ttl: u64,
    ) -> CachedResult {
        CachedResult::new(hash, endpoint, rows, cached_at, ttl)
    }

    // --- CachedResult ----------------------------------------------------

    #[test]
    fn test_cached_result_not_expired() {
        let r = make_result(1, "http://ep1", 1000, 5000);
        assert!(!r.is_expired(3000));
    }

    #[test]
    fn test_cached_result_expired() {
        let r = make_result(1, "http://ep1", 1000, 500);
        assert!(r.is_expired(1600)); // 1600 - 1000 = 600 >= 500
    }

    #[test]
    fn test_cached_result_exactly_at_boundary() {
        let r = make_result(1, "http://ep1", 1000, 500);
        assert!(r.is_expired(1500)); // 1500 - 1000 = 500 >= 500
    }

    #[test]
    fn test_cached_result_clone() {
        let r = make_result(42, "http://ep", 0, 1000);
        let c = r.clone();
        assert_eq!(c.query_hash, 42);
        assert_eq!(c.endpoint, "http://ep");
    }

    // --- ResultCache::new -----------------------------------------------

    #[test]
    fn test_new_empty() {
        let cache = ResultCache::new(10);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_new_zero_size_uses_one() {
        let cache = ResultCache::new(0);
        // max_size is forced to 1
        assert_eq!(cache.max_size, 1);
    }

    // --- insert + get ----------------------------------------------------

    #[test]
    fn test_insert_and_get_hit() {
        let mut cache = ResultCache::new(10);
        let now = 1000u64;
        cache.insert(make_result(1, "http://ep1", now, 5000));
        let r = cache.get(1, now + 100);
        assert!(r.is_some());
    }

    #[test]
    fn test_get_miss() {
        let mut cache = ResultCache::new(10);
        let r = cache.get(99, 1000);
        assert!(r.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_get_expired_returns_none() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "http://ep1", 1000, 100));
        let r = cache.get(1, 2000); // expired
        assert!(r.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_hit_increments_stats() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "http://ep1", 0, 99999));
        cache.get(1, 100);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_get_returns_rows() {
        let mut cache = ResultCache::new(10);
        let mut row = HashMap::new();
        row.insert("?s".into(), "http://example.org/s1".into());
        cache.insert(make_result_rows(5, "http://ep", vec![row.clone()], 0, 99999));
        let r = cache.get(5, 0).expect("hit");
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0]["?s"], "http://example.org/s1");
    }

    // --- LRU eviction ----------------------------------------------------

    #[test]
    fn test_lru_eviction() {
        let mut cache = ResultCache::new(2);
        cache.insert(make_result(1, "ep", 0, 99999));
        cache.insert(make_result(2, "ep", 0, 99999));
        // Access hash=1 to make it MRU
        cache.get(1, 0);
        // Insert hash=3 should evict LRU = hash=2
        cache.insert(make_result(3, "ep", 0, 99999));
        assert!(cache.get(1, 0).is_some(), "hash 1 should still be cached");
        assert!(cache.get(2, 0).is_none(), "hash 2 should have been evicted");
        assert!(cache.get(3, 0).is_some(), "hash 3 should be cached");
    }

    #[test]
    fn test_eviction_increments_stats() {
        let mut cache = ResultCache::new(1);
        cache.insert(make_result(1, "ep", 0, 99999));
        cache.insert(make_result(2, "ep", 0, 99999));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_update_existing_entry() {
        let mut cache = ResultCache::new(5);
        cache.insert(make_result(1, "ep", 0, 100));
        // Replace with a fresh entry
        cache.insert(make_result(1, "ep", 5000, 99999));
        let r = cache.get(1, 5001).expect("should be hit");
        assert_eq!(r.cached_at_ms, 5000);
    }

    // --- evict_expired ---------------------------------------------------

    #[test]
    fn test_evict_expired_removes_old() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "ep", 0, 100));
        cache.insert(make_result(2, "ep", 0, 99999));
        let removed = cache.evict_expired(200);
        assert_eq!(removed, 1);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_evict_expired_none_expired() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "ep", 0, 99999));
        let removed = cache.evict_expired(100);
        assert_eq!(removed, 0);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_evict_expired_all() {
        let mut cache = ResultCache::new(10);
        for i in 0..5u64 {
            cache.insert(make_result(i, "ep", 0, 50));
        }
        let removed = cache.evict_expired(100);
        assert_eq!(removed, 5);
        assert_eq!(cache.size(), 0);
    }

    // --- invalidate_endpoint ---------------------------------------------

    #[test]
    fn test_invalidate_endpoint() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "http://ep1", 0, 99999));
        cache.insert(make_result(2, "http://ep1", 0, 99999));
        cache.insert(make_result(3, "http://ep2", 0, 99999));
        let removed = cache.invalidate_endpoint("http://ep1");
        assert_eq!(removed, 2);
        assert_eq!(cache.size(), 1);
        assert!(cache.get(3, 0).is_some());
    }

    #[test]
    fn test_invalidate_endpoint_not_found() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "http://ep1", 0, 99999));
        let removed = cache.invalidate_endpoint("http://ep_none");
        assert_eq!(removed, 0);
        assert_eq!(cache.size(), 1);
    }

    // --- clear -----------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut cache = ResultCache::new(10);
        for i in 0..5u64 {
            cache.insert(make_result(i, "ep", 0, 99999));
        }
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    // --- hit_rate --------------------------------------------------------

    #[test]
    fn test_hit_rate_no_accesses() {
        let cache = ResultCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "ep", 0, 99999));
        cache.get(1, 0);
        cache.get(1, 0);
        assert!((cache.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_hit_rate_half() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "ep", 0, 99999));
        cache.get(1, 0); // hit
        cache.get(99, 0); // miss
        assert!((cache.hit_rate() - 0.5).abs() < 1e-9);
    }

    // --- hash_query ------------------------------------------------------

    #[test]
    fn test_hash_query_deterministic() {
        let h1 = ResultCache::hash_query("SELECT * WHERE {?s ?p ?o}", "http://ep1");
        let h2 = ResultCache::hash_query("SELECT * WHERE {?s ?p ?o}", "http://ep1");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_query_different_queries() {
        let h1 = ResultCache::hash_query("SELECT * WHERE {?s ?p ?o}", "http://ep1");
        let h2 = ResultCache::hash_query("SELECT ?s WHERE {?s ?p ?o}", "http://ep1");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_query_different_endpoints() {
        let h1 = ResultCache::hash_query("SELECT * WHERE {?s ?p ?o}", "http://ep1");
        let h2 = ResultCache::hash_query("SELECT * WHERE {?s ?p ?o}", "http://ep2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_query_empty_strings() {
        let h = ResultCache::hash_query("", "");
        // Should not panic and produce the FNV offset
        assert_ne!(h, 0);
    }

    // --- Default ---------------------------------------------------------

    #[test]
    fn test_default_cache_size() {
        let cache = ResultCache::default();
        assert_eq!(cache.max_size, 256);
    }

    // --- Misc ------------------------------------------------------------

    #[test]
    fn test_size_after_operations() {
        let mut cache = ResultCache::new(5);
        assert_eq!(cache.size(), 0);
        cache.insert(make_result(1, "ep", 0, 99999));
        assert_eq!(cache.size(), 1);
        cache.insert(make_result(2, "ep", 0, 99999));
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_stats_size_tracks_entries() {
        let mut cache = ResultCache::new(10);
        cache.insert(make_result(1, "ep", 0, 99999));
        cache.insert(make_result(2, "ep", 0, 99999));
        assert_eq!(cache.stats().size, 2);
    }
}
