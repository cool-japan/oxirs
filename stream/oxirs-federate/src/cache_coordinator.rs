//! Distributed query result cache coordinator.
//!
//! Provides an in-memory cache for SPARQL query results with configurable
//! eviction policies (LRU, LFU, FIFO, TTL-only), TTL-based expiry,
//! prefix-based invalidation, and detailed hit/miss/eviction statistics.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Error ──────────────────────────────────────────────────────────────────────

/// Errors produced by the cache coordinator.
#[derive(Debug)]
pub enum CacheError {
    /// The requested operation would exceed the capacity limit.
    CapacityExceeded(String),
    /// A zero or otherwise invalid TTL was supplied.
    InvalidTtl(String),
    /// The requested key was not found.
    KeyNotFound(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::CapacityExceeded(msg) => write!(f, "capacity exceeded: {msg}"),
            CacheError::InvalidTtl(msg) => write!(f, "invalid TTL: {msg}"),
            CacheError::KeyNotFound(key) => write!(f, "key not found: {key}"),
        }
    }
}

impl std::error::Error for CacheError {}

// ── Entry ──────────────────────────────────────────────────────────────────────

/// A single cached query result.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cache key (typically `endpoint:hash_of_query`).
    pub key: String,
    /// Serialised query result bytes.
    pub value: Vec<u8>,
    /// When the entry was inserted.
    pub created_at: Instant,
    /// How long the entry lives before it is considered expired.
    pub ttl: Duration,
    /// Number of successful cache hits for this entry.
    pub hit_count: u64,
    /// Approximate size of `value` in bytes.
    pub size_bytes: usize,
}

impl CacheEntry {
    /// Returns `true` when the entry's age exceeds its TTL.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }

    /// Returns how long ago the entry was created.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ── Stats ──────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the cache coordinator.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Total successful lookups.
    pub hits: u64,
    /// Total unsuccessful lookups.
    pub misses: u64,
    /// Total entries evicted (capacity or explicit).
    pub evictions: u64,
    /// Total bytes currently held in the cache.
    pub total_size_bytes: usize,
    /// Number of entries currently in the cache.
    pub entry_count: usize,
}

impl CacheStats {
    /// `hits / (hits + misses)`, or `0.0` when no requests have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ── Eviction policy ────────────────────────────────────────────────────────────

/// Determines which entry to evict when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the least recently *used* entry.
    Lru,
    /// Evict the least frequently *used* entry.
    Lfu,
    /// Expire-only — no active eviction until `get`/`put` is called.
    Ttl,
    /// Evict the earliest *inserted* entry (first-in, first-out).
    Fifo,
}

// ── FNV-1a hash ───────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ── Coordinator ───────────────────────────────────────────────────────────────

/// In-memory query-result cache with pluggable eviction and TTL expiry.
pub struct CacheCoordinator {
    entries: HashMap<String, CacheEntry>,
    policy: EvictionPolicy,
    max_entries: usize,
    max_size_bytes: usize,
    stats: CacheStats,
    /// Insertion order — used by FIFO and as a fallback.
    access_order: Vec<String>,
    /// Per-key access frequency — used by LFU.
    freq_map: HashMap<String, u64>,
}

impl CacheCoordinator {
    /// Create a new coordinator with the given policy and limits.
    ///
    /// Set `max_size_bytes` to `usize::MAX` to disable the byte limit.
    pub fn new(policy: EvictionPolicy, max_entries: usize, max_size_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            policy,
            max_entries,
            max_size_bytes,
            stats: CacheStats::default(),
            access_order: Vec::new(),
            freq_map: HashMap::new(),
        }
    }

    /// Insert or replace an entry.
    ///
    /// Returns `Err(CacheError::InvalidTtl)` for a zero TTL, and
    /// `Err(CacheError::CapacityExceeded)` when the value alone exceeds
    /// `max_size_bytes`.
    pub fn put(&mut self, key: String, value: Vec<u8>, ttl: Duration) -> Result<(), CacheError> {
        if ttl.is_zero() {
            return Err(CacheError::InvalidTtl("TTL must be non-zero".to_string()));
        }

        let size_bytes = value.len();

        if size_bytes > self.max_size_bytes {
            return Err(CacheError::CapacityExceeded(format!(
                "value size {size_bytes} exceeds max_size_bytes {}",
                self.max_size_bytes
            )));
        }

        // Remove old entry if the key already exists so we can re-insert cleanly.
        if let Some(old) = self.entries.remove(&key) {
            self.stats.total_size_bytes =
                self.stats.total_size_bytes.saturating_sub(old.size_bytes);
            self.access_order.retain(|k| k != &key);
            self.freq_map.remove(&key);
        }

        // Evict until we have room (entries and bytes).
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.stats.total_size_bytes + size_bytes > self.max_size_bytes)
        {
            self.evict_one();
        }

        let entry = CacheEntry {
            key: key.clone(),
            value,
            created_at: Instant::now(),
            ttl,
            hit_count: 0,
            size_bytes,
        };

        self.stats.total_size_bytes += size_bytes;
        self.access_order.push(key.clone());
        self.freq_map.insert(key.clone(), 0);
        self.entries.insert(key, entry);
        self.stats.entry_count = self.entries.len();
        Ok(())
    }

    /// Look up a key, updating hit/miss statistics and access tracking.
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        // First check expiry without borrowing `self.entries` mutably.
        let expired = self
            .entries
            .get(key)
            .map(|e| e.is_expired())
            .unwrap_or(false);

        if expired {
            self.remove_key(key);
            self.stats.misses += 1;
            return None;
        }

        if let Some(entry) = self.entries.get_mut(key) {
            entry.hit_count += 1;
            self.stats.hits += 1;

            // Update LRU order.
            let key_owned = key.to_string();
            self.access_order.retain(|k| k != &key_owned);
            self.access_order.push(key_owned.clone());

            // Update LFU frequency.
            *self.freq_map.entry(key_owned).or_insert(0) += 1;

            // SAFETY: we need a re-borrow to return an immutable reference.
            return self.entries.get(key);
        }

        self.stats.misses += 1;
        None
    }

    /// Remove an entry, returning whether it was present.
    pub fn invalidate(&mut self, key: &str) -> bool {
        let existed = self.remove_key(key);
        if existed {
            self.stats.evictions += 1;
        }
        existed
    }

    /// Remove all entries whose key starts with `prefix`, returning the count.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let matching: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = matching.len();
        for k in matching {
            self.remove_key(&k);
            self.stats.evictions += 1;
        }
        count
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Current number of entries in the cache.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Total bytes currently held in the cache.
    pub fn total_size_bytes(&self) -> usize {
        self.stats.total_size_bytes
    }

    /// Evict all entries that have expired, returning the count removed.
    pub fn evict_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for k in expired_keys {
            self.remove_key(&k);
            self.stats.evictions += 1;
        }
        count
    }

    /// Remove all entries, reset statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.freq_map.clear();
        self.stats = CacheStats::default();
    }

    /// Build a canonical cache key from an endpoint URL and a SPARQL query.
    pub fn query_cache_key(endpoint: &str, query: &str) -> String {
        format!("{}:{:016x}", endpoint, simple_hash(query))
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    /// Evict exactly one entry according to the configured policy.
    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let victim_key = match self.policy {
            EvictionPolicy::Lru | EvictionPolicy::Ttl => {
                // LRU: evict the entry at the front of access_order (least recently used).
                self.access_order.first().cloned()
            }
            EvictionPolicy::Fifo => {
                // FIFO: evict the first inserted entry.
                self.access_order.first().cloned()
            }
            EvictionPolicy::Lfu => {
                // LFU: find the entry with the lowest accumulated frequency.
                self.freq_map
                    .iter()
                    .filter(|(k, _)| self.entries.contains_key(*k))
                    .min_by_key(|(_, &freq)| freq)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(k) = victim_key {
            self.remove_key(&k);
            self.stats.evictions += 1;
        }
    }

    /// Remove a key from all internal structures, returning `true` if it existed.
    fn remove_key(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.stats.total_size_bytes =
                self.stats.total_size_bytes.saturating_sub(entry.size_bytes);
            self.access_order.retain(|k| k != key);
            self.freq_map.remove(key);
            self.stats.entry_count = self.entries.len();
            true
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn make_cache(policy: EvictionPolicy) -> CacheCoordinator {
        CacheCoordinator::new(policy, 10, 1 << 20)
    }

    fn entry_value(n: u8) -> Vec<u8> {
        vec![n; 16]
    }

    // ── put / get ──────────────────────────────────────────────────────────────

    #[test]
    fn test_put_and_get() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k1".into(), entry_value(1), secs(60)).unwrap();
        let e = c.get("k1").expect("should be present");
        assert_eq!(e.value, entry_value(1));
    }

    #[test]
    fn test_put_overwrites_existing_key() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k1".into(), vec![1; 4], secs(60)).unwrap();
        c.put("k1".into(), vec![2; 4], secs(60)).unwrap();
        assert_eq!(c.entry_count(), 1);
        assert_eq!(c.get("k1").unwrap().value, vec![2; 4]);
    }

    #[test]
    fn test_get_missing_key_returns_none() {
        let mut c = make_cache(EvictionPolicy::Lru);
        assert!(c.get("missing").is_none());
    }

    // ── statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_hit_increments_stats() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k1".into(), entry_value(1), secs(60)).unwrap();
        c.get("k1");
        c.get("k1");
        assert_eq!(c.stats().hits, 2);
        assert_eq!(c.stats().misses, 0);
    }

    #[test]
    fn test_cache_miss_increments_stats() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.get("absent");
        c.get("also_absent");
        assert_eq!(c.stats().hits, 0);
        assert_eq!(c.stats().misses, 2);
    }

    #[test]
    fn test_hit_rate_no_requests_is_zero() {
        let c = make_cache(EvictionPolicy::Lru);
        assert_eq!(c.stats().hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), entry_value(1), secs(60)).unwrap();
        c.get("k");
        c.get("k");
        assert!((c.stats().hit_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_mixed() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), entry_value(1), secs(60)).unwrap();
        c.get("k"); // hit
        c.get("absent"); // miss
        let rate = c.stats().hit_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    // ── TTL / expiry ─────────────────────────────────────────────────────────

    #[test]
    fn test_cache_entry_is_expired_when_ttl_elapsed() {
        // Build an entry whose `created_at` is far in the past by subtracting time.
        let entry = CacheEntry {
            key: "k".into(),
            value: vec![],
            created_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(1),
            hit_count: 0,
            size_bytes: 0,
        };
        assert!(entry.is_expired());
    }

    #[test]
    fn test_cache_entry_not_expired_within_ttl() {
        let entry = CacheEntry {
            key: "k".into(),
            value: vec![],
            created_at: Instant::now(),
            ttl: Duration::from_secs(3600),
            hit_count: 0,
            size_bytes: 0,
        };
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_zero_ttl_rejected() {
        let mut c = make_cache(EvictionPolicy::Lru);
        let result = c.put("k".into(), vec![], Duration::ZERO);
        assert!(result.is_err());
        assert!(matches!(result, Err(CacheError::InvalidTtl(_))));
    }

    #[test]
    fn test_evict_expired_removes_old_entries() {
        let mut c = make_cache(EvictionPolicy::Ttl);

        // Insert an already-expired entry by crafting CacheEntry directly.
        let expired_entry = CacheEntry {
            key: "old".into(),
            value: vec![9; 8],
            created_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(1),
            hit_count: 0,
            size_bytes: 8,
        };
        c.entries.insert("old".into(), expired_entry);
        c.access_order.push("old".into());
        c.freq_map.insert("old".into(), 0);
        c.stats.total_size_bytes += 8;
        c.stats.entry_count = 1;

        // Also insert a live entry.
        c.put("live".into(), vec![1; 8], Duration::from_secs(3600))
            .unwrap();

        let removed = c.evict_expired();
        assert_eq!(removed, 1);
        assert!(c.get("old").is_none());
        assert!(c.get("live").is_some());
    }

    #[test]
    fn test_get_expired_entry_returns_none() {
        let mut c = make_cache(EvictionPolicy::Ttl);

        let expired_entry = CacheEntry {
            key: "ex".into(),
            value: vec![1; 4],
            created_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(1),
            hit_count: 0,
            size_bytes: 4,
        };
        c.entries.insert("ex".into(), expired_entry);
        c.access_order.push("ex".into());
        c.freq_map.insert("ex".into(), 0);
        c.stats.total_size_bytes += 4;
        c.stats.entry_count = 1;

        assert!(c.get("ex").is_none());
        assert_eq!(c.stats().misses, 1);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_when_max_entries_exceeded() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Lru, 3, 1 << 20);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.put("c".into(), vec![3], secs(60)).unwrap();

        // Access 'a' to make it recently used; 'b' becomes LRU.
        c.get("a");
        c.get("c");

        // This put should evict 'b'.
        c.put("d".into(), vec![4], secs(60)).unwrap();

        assert!(c.get("b").is_none(), "b should have been evicted");
        assert!(c.get("a").is_some());
        assert!(c.get("c").is_some());
        assert!(c.get("d").is_some());
    }

    #[test]
    fn test_lru_eviction_increments_eviction_count() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Lru, 2, 1 << 20);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.put("c".into(), vec![3], secs(60)).unwrap(); // evicts 'a'
        assert_eq!(c.stats().evictions, 1);
    }

    // ── FIFO eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_fifo_eviction_removes_oldest_insert() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Fifo, 3, 1 << 20);
        c.put("first".into(), vec![1], secs(60)).unwrap();
        c.put("second".into(), vec![2], secs(60)).unwrap();
        c.put("third".into(), vec![3], secs(60)).unwrap();
        c.put("fourth".into(), vec![4], secs(60)).unwrap(); // evicts 'first'
        assert!(c.get("first").is_none());
        assert!(c.get("second").is_some());
        assert!(c.get("fourth").is_some());
    }

    // ── LFU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_lfu_eviction_removes_least_frequently_used() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Lfu, 3, 1 << 20);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.put("c".into(), vec![3], secs(60)).unwrap();
        // Access 'a' and 'c' multiple times; 'b' has frequency 0.
        c.get("a");
        c.get("a");
        c.get("c");
        // Next put evicts 'b' (lowest frequency).
        c.put("d".into(), vec![4], secs(60)).unwrap();
        assert!(
            c.get("b").is_none(),
            "b should have been evicted (lowest freq)"
        );
        assert!(c.get("a").is_some());
        assert!(c.get("d").is_some());
    }

    // ── invalidate ───────────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_present_key() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), vec![1], secs(60)).unwrap();
        assert!(c.invalidate("k"));
        assert!(c.get("k").is_none());
    }

    #[test]
    fn test_invalidate_absent_key_returns_false() {
        let mut c = make_cache(EvictionPolicy::Lru);
        assert!(!c.invalidate("ghost"));
    }

    #[test]
    fn test_invalidate_decreases_entry_count() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("x".into(), vec![1], secs(60)).unwrap();
        c.put("y".into(), vec![2], secs(60)).unwrap();
        c.invalidate("x");
        assert_eq!(c.entry_count(), 1);
    }

    // ── invalidate_prefix ────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_prefix_removes_matching_keys() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("ep1:q1".into(), vec![1], secs(60)).unwrap();
        c.put("ep1:q2".into(), vec![2], secs(60)).unwrap();
        c.put("ep2:q1".into(), vec![3], secs(60)).unwrap();
        let count = c.invalidate_prefix("ep1:");
        assert_eq!(count, 2);
        assert!(c.get("ep1:q1").is_none());
        assert!(c.get("ep1:q2").is_none());
        assert!(c.get("ep2:q1").is_some());
    }

    #[test]
    fn test_invalidate_prefix_no_match_returns_zero() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), vec![1], secs(60)).unwrap();
        assert_eq!(c.invalidate_prefix("nomatch:"), 0);
    }

    // ── size tracking ─────────────────────────────────────────────────────────

    #[test]
    fn test_size_tracking_on_insert() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), vec![0u8; 100], secs(60)).unwrap();
        assert_eq!(c.total_size_bytes(), 100);
    }

    #[test]
    fn test_size_decreases_on_invalidate() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), vec![0u8; 100], secs(60)).unwrap();
        c.invalidate("k");
        assert_eq!(c.total_size_bytes(), 0);
    }

    #[test]
    fn test_max_size_bytes_overflow_rejected() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Lru, 100, 50);
        let result = c.put("big".into(), vec![0u8; 100], secs(60));
        assert!(matches!(result, Err(CacheError::CapacityExceeded(_))));
    }

    #[test]
    fn test_size_bytes_tracks_multiple_entries() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("a".into(), vec![0u8; 30], secs(60)).unwrap();
        c.put("b".into(), vec![0u8; 70], secs(60)).unwrap();
        assert_eq!(c.total_size_bytes(), 100);
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_removes_all_entries() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.clear();
        assert_eq!(c.entry_count(), 0);
        assert_eq!(c.total_size_bytes(), 0);
    }

    #[test]
    fn test_clear_resets_stats() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.get("a");
        c.clear();
        let s = c.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.total_size_bytes, 0);
        assert_eq!(s.entry_count, 0);
    }

    // ── query_cache_key ───────────────────────────────────────────────────────

    #[test]
    fn test_query_cache_key_deterministic() {
        let k1 = CacheCoordinator::query_cache_key("http://ep/", "SELECT * WHERE { ?s ?p ?o }");
        let k2 = CacheCoordinator::query_cache_key("http://ep/", "SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_different_queries() {
        let k1 = CacheCoordinator::query_cache_key("http://ep/", "SELECT ?s WHERE { ?s a ?t }");
        let k2 = CacheCoordinator::query_cache_key("http://ep/", "SELECT ?p WHERE { ?s ?p ?o }");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_different_endpoints() {
        let k1 = CacheCoordinator::query_cache_key("http://ep1/", "SELECT * {}");
        let k2 = CacheCoordinator::query_cache_key("http://ep2/", "SELECT * {}");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_query_cache_key_format_contains_endpoint() {
        let k = CacheCoordinator::query_cache_key("http://ep/sparql", "ASK {}");
        assert!(k.starts_with("http://ep/sparql:"));
    }

    // ── hit_count on entry ────────────────────────────────────────────────────

    #[test]
    fn test_hit_count_increments_on_each_get() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("k".into(), vec![1], secs(60)).unwrap();
        c.get("k");
        c.get("k");
        c.get("k");
        // Re-fetch via get to observe the latest hit_count.
        let entry = c.entries.get("k").unwrap();
        assert_eq!(entry.hit_count, 3);
    }

    // ── entry_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_entry_count_tracks_insertions() {
        let mut c = make_cache(EvictionPolicy::Lru);
        assert_eq!(c.entry_count(), 0);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        assert_eq!(c.entry_count(), 1);
        c.put("b".into(), vec![2], secs(60)).unwrap();
        assert_eq!(c.entry_count(), 2);
    }

    #[test]
    fn test_entry_count_decreases_on_invalidate() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.invalidate("a");
        assert_eq!(c.entry_count(), 0);
    }

    // ── age ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_entry_age_is_non_negative() {
        let entry = CacheEntry {
            key: "k".into(),
            value: vec![],
            created_at: Instant::now(),
            ttl: secs(60),
            hit_count: 0,
            size_bytes: 0,
        };
        assert!(entry.age() < Duration::from_secs(1));
    }

    // ── evict_expired with multiple expired ────────────────────────────────────

    #[test]
    fn test_evict_expired_multiple() {
        let mut c = make_cache(EvictionPolicy::Ttl);
        for i in 0..5u8 {
            let key = format!("old{i}");
            let expired = CacheEntry {
                key: key.clone(),
                value: vec![i],
                created_at: Instant::now() - Duration::from_secs(200),
                ttl: Duration::from_secs(1),
                hit_count: 0,
                size_bytes: 1,
            };
            c.entries.insert(key.clone(), expired);
            c.access_order.push(key.clone());
            c.freq_map.insert(key, 0);
            c.stats.total_size_bytes += 1;
        }
        c.stats.entry_count = 5;
        c.put("live".into(), vec![99], secs(3600)).unwrap();
        let removed = c.evict_expired();
        assert_eq!(removed, 5);
        assert_eq!(c.entry_count(), 1);
    }

    // ── simple_hash ───────────────────────────────────────────────────────────

    #[test]
    fn test_simple_hash_deterministic() {
        assert_eq!(simple_hash("hello"), simple_hash("hello"));
    }

    #[test]
    fn test_simple_hash_different_strings() {
        assert_ne!(simple_hash("foo"), simple_hash("bar"));
    }

    #[test]
    fn test_simple_hash_empty_string() {
        let h = simple_hash("");
        assert_eq!(h, 14_695_981_039_346_656_037u64); // FNV offset basis
    }

    // ── LRU ordering detail ───────────────────────────────────────────────────

    #[test]
    fn test_lru_evicts_least_recently_used_not_least_inserted() {
        let mut c = CacheCoordinator::new(EvictionPolicy::Lru, 3, 1 << 20);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.put("c".into(), vec![3], secs(60)).unwrap();

        // Touch 'a' so 'b' becomes LRU.
        c.get("c");
        c.get("a");

        c.put("d".into(), vec![4], secs(60)).unwrap();
        assert!(c.get("b").is_none(), "b should be evicted (LRU)");
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_cache_error_display_capacity_exceeded() {
        let e = CacheError::CapacityExceeded("too big".into());
        assert!(e.to_string().contains("too big"));
    }

    #[test]
    fn test_cache_error_display_invalid_ttl() {
        let e = CacheError::InvalidTtl("zero".into());
        assert!(e.to_string().contains("zero"));
    }

    #[test]
    fn test_cache_error_display_key_not_found() {
        let e = CacheError::KeyNotFound("mykey".into());
        assert!(e.to_string().contains("mykey"));
    }

    // ── prefix invalidation edge cases ────────────────────────────────────────

    #[test]
    fn test_invalidate_full_key_as_prefix() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("abc".into(), vec![1], secs(60)).unwrap();
        c.put("abcd".into(), vec![2], secs(60)).unwrap();
        let count = c.invalidate_prefix("abc");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_invalidate_prefix_empty_prefix_removes_all() {
        let mut c = make_cache(EvictionPolicy::Lru);
        c.put("a".into(), vec![1], secs(60)).unwrap();
        c.put("b".into(), vec![2], secs(60)).unwrap();
        c.put("c".into(), vec![3], secs(60)).unwrap();
        let count = c.invalidate_prefix("");
        assert_eq!(count, 3);
        assert_eq!(c.entry_count(), 0);
    }

    // ── size eviction (byte limit) ────────────────────────────────────────────

    #[test]
    fn test_eviction_triggered_by_size_limit() {
        // max_size_bytes = 100; each entry is 60 bytes.
        // After inserting 'a' (60 B), inserting 'b' (60 B) would exceed 100 B → evict 'a'.
        let mut c = CacheCoordinator::new(EvictionPolicy::Lru, 100, 100);
        c.put("a".into(), vec![0u8; 60], secs(60)).unwrap();
        c.put("b".into(), vec![0u8; 60], secs(60)).unwrap();
        assert!(c.get("a").is_none(), "a should have been evicted");
        assert!(c.get("b").is_some());
    }
}
