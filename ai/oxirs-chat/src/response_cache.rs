//! # RAG Response Cache
//!
//! An in-memory LRU cache for RAG (Retrieval-Augmented Generation) responses with:
//!
//! - **TTL-based expiry**: entries older than their TTL are transparently ignored.
//! - **LRU eviction**: when the cache is full, the least-recently-used entry is dropped.
//! - **Semantic-key hashing**: queries are hashed with FNV-1a so the key is a fixed-size `u64`.
//! - **Hit/miss statistics**: monotonically incrementing counters for observability.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_chat::response_cache::ResponseCache;
//!
//! let mut cache = ResponseCache::new(100, 60_000); // capacity=100, ttl=1 min
//! cache.put("What is RDF?", "RDF is a data model.".to_string(), 0);
//! assert_eq!(
//!     cache.get("What is RDF?", 30_000),
//!     Some("RDF is a data model.".to_string())
//! );
//! assert!(cache.get("What is RDF?", 70_000).is_none()); // expired
//! ```

use std::collections::{HashMap, VecDeque};

// ─── FNV-1a hash ──────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of a byte slice (no external crate required).
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the cache key hash for a query string.
fn query_hash(query: &str) -> u64 {
    fnv1a_hash(query.as_bytes())
}

// ─── Data structures ──────────────────────────────────────────────────────────

/// A single entry stored in the response cache.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// FNV-1a hash of the original query string.
    pub query_hash: u64,
    /// The cached response text.
    pub response: String,
    /// Absolute timestamp (ms since epoch or monotonic reference) when stored.
    pub created_at: u64,
    /// Number of times this entry has been successfully retrieved.
    pub access_count: usize,
    /// Lifetime of this entry in milliseconds.
    pub ttl_ms: u64,
}

impl CacheEntry {
    /// Returns `true` if the entry has not yet expired relative to `now_ms`.
    pub fn is_valid(&self, now_ms: u64) -> bool {
        now_ms < self.created_at.saturating_add(self.ttl_ms)
    }
}

/// Snapshot of cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of unsuccessful cache lookups (including expired entries).
    pub misses: u64,
    /// Number of entries evicted to make room for new ones.
    pub evictions: u64,
    /// Current number of live (potentially expired) entries in the cache.
    pub entries: usize,
}

// ─── ResponseCache ────────────────────────────────────────────────────────────

/// LRU cache for RAG responses with per-entry TTL.
///
/// Internally uses:
/// - A `HashMap<u64, CacheEntry>` for O(1) lookup by hash.
/// - A `VecDeque<u64>` that records insertion/access order for LRU eviction.
pub struct ResponseCache {
    capacity: usize,
    default_ttl_ms: u64,
    entries: HashMap<u64, CacheEntry>,
    /// Front = most recently used, back = least recently used.
    lru_order: VecDeque<u64>,
    stats: CacheStats,
}

impl ResponseCache {
    /// Create a new cache with the given `capacity` (maximum number of entries)
    /// and `default_ttl_ms` (entry lifetime in milliseconds).
    pub fn new(capacity: usize, default_ttl_ms: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            default_ttl_ms,
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Look up a cached response for `query` at the given timestamp.
    ///
    /// - Returns `Some(response)` if an unexpired entry exists and updates
    ///   `access_count` and the hit counter.
    /// - Returns `None` (and increments the miss counter) otherwise.
    pub fn get(&mut self, query: &str, now_ms: u64) -> Option<String> {
        let key = query_hash(query);

        match self.entries.get_mut(&key) {
            Some(entry) if entry.is_valid(now_ms) => {
                entry.access_count += 1;
                self.stats.hits += 1;
                let response = entry.response.clone();
                // Move to front of LRU queue.
                self.touch_lru(key);
                Some(response)
            }
            _ => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert or replace a cache entry for `query`.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted first.
    pub fn put(&mut self, query: &str, response: String, now_ms: u64) {
        let key = query_hash(query);

        // Evict if at capacity and this is a new key.
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        // Remove old LRU position if key already exists.
        if self.entries.contains_key(&key) {
            self.lru_order.retain(|&k| k != key);
        }

        let entry = CacheEntry {
            query_hash: key,
            response,
            created_at: now_ms,
            access_count: 0,
            ttl_ms: self.default_ttl_ms,
        };

        self.entries.insert(key, entry);
        self.lru_order.push_front(key);
        self.stats.entries = self.entries.len();
    }

    /// Remove the entry for `query` (if any).
    ///
    /// Returns `true` if an entry was actually removed.
    pub fn invalidate(&mut self, query: &str) -> bool {
        let key = query_hash(query);
        if self.entries.remove(&key).is_some() {
            self.lru_order.retain(|&k| k != key);
            self.stats.entries = self.entries.len();
            true
        } else {
            false
        }
    }

    /// Remove all entries whose TTL has elapsed relative to `now_ms`.
    ///
    /// Returns the number of entries that were purged.
    pub fn invalidate_expired(&mut self, now_ms: u64) -> usize {
        let expired_keys: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.is_valid(now_ms))
            .map(|(k, _)| *k)
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.entries.remove(&key);
            self.lru_order.retain(|&k| k != key);
        }

        self.stats.entries = self.entries.len();
        count
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Current number of stored entries (including potentially expired ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries the cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The default TTL applied to new entries.
    pub fn default_ttl_ms(&self) -> u64 {
        self.default_ttl_ms
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats = CacheStats::default();
    }

    // ── internal helpers ────────────────────────────────────────────────────

    /// Move `key` to the front of the LRU deque (most-recently-used position).
    fn touch_lru(&mut self, key: u64) {
        self.lru_order.retain(|&k| k != key);
        self.lru_order.push_front(key);
    }

    /// Remove the least-recently-used entry from the cache.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.lru_order.pop_back() {
            self.entries.remove(&lru_key);
            self.stats.evictions += 1;
            self.stats.entries = self.entries.len();
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FNV-1a hash ──────────────────────────────────────────────────────────

    #[test]
    fn test_hash_deterministic() {
        assert_eq!(query_hash("hello"), query_hash("hello"));
    }

    #[test]
    fn test_hash_different_inputs() {
        assert_ne!(query_hash("hello"), query_hash("world"));
    }

    #[test]
    fn test_hash_empty_string() {
        let h = query_hash("");
        // Should not panic and should return the FNV offset basis (no bytes mixed).
        let _ = h;
    }

    // ── CacheEntry helpers ───────────────────────────────────────────────────

    #[test]
    fn test_entry_is_valid_before_expiry() {
        let entry = CacheEntry {
            query_hash: 1,
            response: "r".to_string(),
            created_at: 1000,
            access_count: 0,
            ttl_ms: 5000,
        };
        assert!(entry.is_valid(5999));
    }

    #[test]
    fn test_entry_is_invalid_after_expiry() {
        let entry = CacheEntry {
            query_hash: 1,
            response: "r".to_string(),
            created_at: 1000,
            access_count: 0,
            ttl_ms: 5000,
        };
        assert!(!entry.is_valid(6001));
    }

    #[test]
    fn test_entry_exact_boundary_invalid() {
        let entry = CacheEntry {
            query_hash: 1,
            response: "r".to_string(),
            created_at: 1000,
            access_count: 0,
            ttl_ms: 5000,
        };
        // now == created_at + ttl_ms  => NOT valid (strict <)
        assert!(!entry.is_valid(6000));
    }

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_cache_is_empty() {
        let cache = ResponseCache::new(100, 60_000);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_new_cache_capacity_stored() {
        let cache = ResponseCache::new(50, 10_000);
        assert_eq!(cache.capacity(), 50);
    }

    #[test]
    fn test_new_cache_ttl_stored() {
        let cache = ResponseCache::new(50, 99_999);
        assert_eq!(cache.default_ttl_ms(), 99_999);
    }

    // ── put / get ────────────────────────────────────────────────────────────

    #[test]
    fn test_basic_put_and_get() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q1", "response1".to_string(), 0);
        assert_eq!(cache.get("q1", 1000), Some("response1".to_string()));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let mut cache = ResponseCache::new(10, 60_000);
        assert!(cache.get("nonexistent", 0).is_none());
    }

    #[test]
    fn test_get_after_ttl_expiry_returns_none() {
        let mut cache = ResponseCache::new(10, 5_000);
        cache.put("q", "r".to_string(), 0);
        assert!(cache.get("q", 5_001).is_none());
    }

    #[test]
    fn test_get_within_ttl_succeeds() {
        let mut cache = ResponseCache::new(10, 5_000);
        cache.put("q", "r".to_string(), 1000);
        assert!(cache.get("q", 5_999).is_some());
    }

    #[test]
    fn test_put_updates_existing_entry() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "v1".to_string(), 0);
        cache.put("q", "v2".to_string(), 0);
        assert_eq!(cache.get("q", 0), Some("v2".to_string()));
        assert_eq!(cache.len(), 1); // still one entry
    }

    #[test]
    fn test_put_multiple_entries() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        cache.put("q3", "r3".to_string(), 0);
        assert_eq!(cache.len(), 3);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_at_capacity() {
        let mut cache = ResponseCache::new(3, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        cache.put("q3", "r3".to_string(), 0);
        // Access q1 to make q2 the LRU
        cache.get("q1", 0);
        // Insert q4 — should evict q2 (LRU)
        cache.put("q4", "r4".to_string(), 0);
        assert_eq!(cache.len(), 3);
        assert!(cache.get("q2", 0).is_none()); // evicted
        assert!(cache.get("q1", 0).is_some()); // not evicted
        assert!(cache.get("q3", 0).is_some()); // not evicted
        assert!(cache.get("q4", 0).is_some()); // not evicted
    }

    #[test]
    fn test_eviction_counter_increments() {
        let mut cache = ResponseCache::new(2, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        cache.put("q3", "r3".to_string(), 0); // evicts one
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_no_eviction_within_capacity() {
        let mut cache = ResponseCache::new(5, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        assert_eq!(cache.stats().evictions, 0);
    }

    // ── invalidate ───────────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_existing_returns_true() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        assert!(cache.invalidate("q"));
    }

    #[test]
    fn test_invalidate_missing_returns_false() {
        let mut cache = ResponseCache::new(10, 60_000);
        assert!(!cache.invalidate("nonexistent"));
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        cache.invalidate("q");
        assert!(cache.get("q", 0).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_invalidate_allows_reinsertion() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r1".to_string(), 0);
        cache.invalidate("q");
        cache.put("q", "r2".to_string(), 0);
        assert_eq!(cache.get("q", 0), Some("r2".to_string()));
    }

    // ── invalidate_expired ───────────────────────────────────────────────────

    #[test]
    fn test_invalidate_expired_removes_stale() {
        let mut cache = ResponseCache::new(10, 1_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        let purged = cache.invalidate_expired(2_000); // both expired
        assert_eq!(purged, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_invalidate_expired_keeps_fresh() {
        let mut cache = ResponseCache::new(10, 10_000);
        cache.put("q1", "r1".to_string(), 0); // expires at 10000
        cache.put("q2", "r2".to_string(), 5_000); // expires at 15000
        let purged = cache.invalidate_expired(11_000); // q1 expired, q2 fresh
        assert_eq!(purged, 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get("q2", 11_000).is_some());
    }

    #[test]
    fn test_invalidate_expired_none_expired() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        let purged = cache.invalidate_expired(1_000);
        assert_eq!(purged, 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_invalidate_expired_empty_cache() {
        let mut cache = ResponseCache::new(10, 60_000);
        assert_eq!(cache.invalidate_expired(0), 0);
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zeros() {
        let cache = ResponseCache::new(10, 60_000);
        let s = cache.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.entries, 0);
    }

    #[test]
    fn test_stats_hit_increments() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        cache.get("q", 0);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_stats_miss_increments() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.get("q", 0);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_stats_miss_on_expired() {
        let mut cache = ResponseCache::new(10, 1_000);
        cache.put("q", "r".to_string(), 0);
        cache.get("q", 2_000); // expired => miss
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_stats_entries_tracks_len() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        assert_eq!(cache.stats().entries, 2);
        cache.invalidate("q1");
        assert_eq!(cache.stats().entries, 1);
    }

    // ── access_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_access_count_increments_on_hit() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        cache.get("q", 0);
        cache.get("q", 0);
        let entry = cache.entries.get(&query_hash("q")).expect("entry exists");
        assert_eq!(entry.access_count, 2);
    }

    // ── len / is_empty ───────────────────────────────────────────────────────

    #[test]
    fn test_is_empty_after_clear() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_len_after_clear() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_stats() {
        let mut cache = ResponseCache::new(10, 60_000);
        cache.put("q", "r".to_string(), 0);
        cache.get("q", 0);
        cache.clear();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().entries, 0);
    }

    // ── capacity guard ───────────────────────────────────────────────────────

    #[test]
    fn test_minimum_capacity_is_one() {
        let cache = ResponseCache::new(0, 60_000);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_large_capacity() {
        let cache = ResponseCache::new(100_000, 3_600_000);
        assert_eq!(cache.capacity(), 100_000);
    }

    // ── LRU ordering after multiple accesses ──────────────────────────────────

    #[test]
    fn test_lru_order_after_access() {
        let mut cache = ResponseCache::new(3, 60_000);
        cache.put("q1", "r1".to_string(), 0);
        cache.put("q2", "r2".to_string(), 0);
        cache.put("q3", "r3".to_string(), 0);
        // Access q1 — now q2 is LRU
        cache.get("q1", 0);
        cache.get("q3", 0);
        // q2 is LRU; inserting q4 should evict q2
        cache.put("q4", "r4".to_string(), 0);
        assert!(cache.get("q2", 0).is_none());
        assert!(cache.get("q1", 0).is_some());
        assert!(cache.get("q3", 0).is_some());
        assert!(cache.get("q4", 0).is_some());
    }

    // ── edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_zero_ttl_always_expired() {
        let mut cache = ResponseCache::new(10, 0);
        cache.put("q", "r".to_string(), 0);
        // now_ms == 0, created_at == 0, ttl == 0 => 0 < 0 is false => expired
        assert!(cache.get("q", 0).is_none());
    }

    #[test]
    fn test_very_large_ttl() {
        let mut cache = ResponseCache::new(10, u64::MAX / 2);
        cache.put("q", "r".to_string(), 0);
        assert!(cache.get("q", u64::MAX / 4).is_some());
    }
}
