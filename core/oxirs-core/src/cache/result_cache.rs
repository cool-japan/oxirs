//! Core-level SPARQL query result cache with LRU eviction and delta-based invalidation.
//!
//! This module provides a thread-safe, dataset-scoped cache for SPARQL query results.  It is
//! designed to sit at the **core layer**, below the individual query engine implementations, so
//! that all engines (ARQ, rule-based, federated, …) can share a single cache.
//!
//! # Features
//!
//! - **FNV-1a fingerprinting**: Fast, collision-resistant query hashing.
//! - **LRU eviction**: Least-recently-used entries are evicted when the capacity is reached.
//! - **TTL expiration**: Entries expire after a configurable time-to-live.
//! - **Dataset-scoped invalidation**: All entries belonging to a dataset can be invalidated in
//!   one call, e.g. after bulk updates.
//! - **Predicate-scoped invalidation**: Only entries that accessed a specific predicate are
//!   invalidated — allowing fine-grained cache management.
//! - **Delta-driven invalidation**: A list of [`TripleDelta`] events is used to determine exactly
//!   which entries are affected by a set of changes.
//! - **Metrics**: Hit/miss counters accessible via atomic loads.
//!
//! # Thread safety
//!
//! All public methods take `&self`; internal state is protected by a `Mutex`.  The design avoids
//! `RwLock` because writes (TTL/access-time updates on cache hit) are almost as frequent as reads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use crate::view::incremental::TripleDelta;

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// A composite cache key that uniquely identifies a cached query result.
///
/// The key combines a dataset identifier with a 64-bit fingerprint of the query text so that the
/// same query run against different datasets is stored in separate entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoreCacheKey {
    /// Identifier of the dataset this query was run against (e.g. a URL or a name).
    pub dataset_id: String,
    /// FNV-1a hash of the normalized query text.
    pub query_fingerprint: u64,
}

impl CoreCacheKey {
    /// Construct a new cache key from a dataset identifier and the raw SPARQL query text.
    pub fn new(dataset_id: &str, query: &str) -> Self {
        Self {
            dataset_id: dataset_id.to_owned(),
            query_fingerprint: Self::fingerprint(query),
        }
    }

    /// FNV-1a hash of `s`.
    ///
    /// FNV-1a is fast, simple, and has good avalanche properties for short string keys.
    fn fingerprint(s: &str) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;

        let mut hash = FNV_OFFSET;
        for byte in s.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// An individual entry in the result cache.
#[derive(Debug, Clone)]
pub struct CoreCacheEntry {
    /// Composite key that identifies this entry.
    pub key: CoreCacheKey,
    /// Cached result rows (variable bindings).
    pub result_rows: Vec<HashMap<String, String>>,
    /// Predicate IRIs that the cached query accessed.  Used for targeted invalidation.
    pub accessed_predicates: Vec<String>,
    /// When this entry was stored.
    pub created_at: Instant,
    /// When this entry was last read.
    pub last_accessed: Instant,
    /// Entry becomes invalid after this instant.
    pub expires_at: Instant,
    /// Number of times this entry has been served from the cache.
    pub hit_count: u64,
}

impl CoreCacheEntry {
    /// Return `true` if this entry has passed its TTL.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Record a cache hit: bump the hit counter and update `last_accessed`.
    fn touch(&mut self) {
        self.hit_count += 1;
        self.last_accessed = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// LRU node
// ---------------------------------------------------------------------------

/// Intrusive doubly-linked LRU list node stored alongside the entry map.
///
/// We use a `Vec<String>` as an ordered access log rather than a true linked list to avoid
/// unsafe pointer arithmetic.  For cache sizes up to a few thousand entries this is fast enough;
/// large-scale deployments can opt for a more sophisticated structure.
struct LruList {
    /// Ordered by access time: front = least recently used, back = most recently used.
    order: Vec<CoreCacheKey>,
}

impl LruList {
    fn new(capacity: usize) -> Self {
        Self {
            order: Vec::with_capacity(capacity),
        }
    }

    /// Record an access for `key`, moving it to the back of the list.
    fn touch(&mut self, key: &CoreCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push(key.clone());
    }

    /// Remove `key` from the list (called on explicit removal/invalidation).
    fn remove(&mut self, key: &CoreCacheKey) {
        self.order.retain(|k| k != key);
    }

    /// Return the least-recently-used key, or `None` if the list is empty.
    fn pop_lru(&mut self) -> Option<CoreCacheKey> {
        if self.order.is_empty() {
            None
        } else {
            Some(self.order.remove(0))
        }
    }

    fn len(&self) -> usize {
        self.order.len()
    }
}

// ---------------------------------------------------------------------------
// Cache internals (held under a single Mutex)
// ---------------------------------------------------------------------------

struct CacheInner {
    entries: HashMap<CoreCacheKey, CoreCacheEntry>,
    lru: LruList,
    capacity: usize,
}

impl CacheInner {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru: LruList::new(capacity),
            capacity,
        }
    }

    /// Evict entries until `entries.len() < capacity`.  Returns the number evicted.
    fn evict_to_capacity(&mut self) -> usize {
        let mut evicted = 0;
        while self.entries.len() >= self.capacity {
            if let Some(lru_key) = self.lru.pop_lru() {
                self.entries.remove(&lru_key);
                evicted += 1;
            } else {
                break;
            }
        }
        evicted
    }

    /// Remove all expired entries and return the count removed.
    fn purge_expired(&mut self) -> usize {
        let expired: Vec<CoreCacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for key in &expired {
            self.entries.remove(key);
            self.lru.remove(key);
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Public cache type
// ---------------------------------------------------------------------------

/// Thread-safe core-level SPARQL query result cache.
///
/// Construct with [`CoreResultCache::new`] and use `put` / `get` for caching.
///
/// # Example
///
/// ```
/// use oxirs_core::cache::result_cache::{CoreResultCache, CoreCacheKey};
/// use std::time::Duration;
///
/// let cache = CoreResultCache::new(1000, Duration::from_secs(300));
/// let key   = CoreCacheKey::new("my_dataset", "SELECT * WHERE { ?s ?p ?o }");
/// cache.put(key.clone(), vec![], vec!["http://p/name".to_string()]);
/// assert!(cache.get(&key).is_some());
/// ```
pub struct CoreResultCache {
    inner: Arc<Mutex<CacheInner>>,
    default_ttl: Duration,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl CoreResultCache {
    /// Create a new cache with `capacity` entries and a default TTL.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner::new(capacity.max(1)))),
            default_ttl: ttl,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Retrieve a cached result.
    ///
    /// Returns `None` on a miss (including expired entries, which are then removed).
    /// On a hit, `last_accessed` and `hit_count` are updated and the entry is moved to the
    /// MRU position.
    pub fn get(&self, key: &CoreCacheKey) -> Option<Vec<HashMap<String, String>>> {
        let mut inner = self.inner.lock().expect("cache lock poisoned");

        if let Some(entry) = inner.entries.get_mut(key) {
            if entry.is_expired() {
                // Remove expired entry and count as a miss.
                let key_clone = key.clone();
                inner.entries.remove(&key_clone);
                inner.lru.remove(&key_clone);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            entry.touch();
            let result = entry.result_rows.clone();
            inner.lru.touch(key);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(result)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store a query result in the cache.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted first.  Expired
    /// entries are also purged opportunistically on every `put`.
    pub fn put(
        &self,
        key: CoreCacheKey,
        rows: Vec<HashMap<String, String>>,
        predicates: Vec<String>,
    ) {
        self.put_with_ttl(key, rows, predicates, self.default_ttl);
    }

    /// Like [`CoreResultCache::put`] but with a custom TTL for this entry.
    pub fn put_with_ttl(
        &self,
        key: CoreCacheKey,
        rows: Vec<HashMap<String, String>>,
        predicates: Vec<String>,
        ttl: Duration,
    ) {
        let now = Instant::now();
        let entry = CoreCacheEntry {
            key: key.clone(),
            result_rows: rows,
            accessed_predicates: predicates,
            created_at: now,
            last_accessed: now,
            expires_at: now + ttl,
            hit_count: 0,
        };

        let mut inner = self.inner.lock().expect("cache lock poisoned");

        // Opportunistically purge expired entries.
        inner.purge_expired();

        // Make room if needed.
        inner.evict_to_capacity();

        // If key already exists, remove it from the LRU list first.
        if inner.entries.contains_key(&key) {
            inner.lru.remove(&key);
        }

        inner.lru.touch(&key);
        inner.entries.insert(key, entry);
    }

    /// Invalidate all entries for a given dataset.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_dataset(&self, dataset_id: &str) -> usize {
        let mut inner = self.inner.lock().expect("cache lock poisoned");

        let to_remove: Vec<CoreCacheKey> = inner
            .entries
            .keys()
            .filter(|k| k.dataset_id == dataset_id)
            .cloned()
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            inner.entries.remove(key);
            inner.lru.remove(key);
        }
        count
    }

    /// Invalidate all entries for a dataset that accessed a specific predicate.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_predicate(&self, dataset_id: &str, predicate: &str) -> usize {
        let mut inner = self.inner.lock().expect("cache lock poisoned");

        let to_remove: Vec<CoreCacheKey> = inner
            .entries
            .iter()
            .filter(|(k, e)| {
                k.dataset_id == dataset_id
                    && (e.accessed_predicates.is_empty()
                        || e.accessed_predicates.iter().any(|p| p == predicate))
            })
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            inner.entries.remove(key);
            inner.lru.remove(key);
        }
        count
    }

    /// Invalidate cache entries affected by the given triple deltas.
    ///
    /// An entry is considered affected if:
    /// - Its `accessed_predicates` list is empty (wildcard), **or**
    /// - Any of the delta predicates appears in the entry's `accessed_predicates`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_on_delta(&self, dataset_id: &str, deltas: &[TripleDelta]) -> usize {
        if deltas.is_empty() {
            return 0;
        }

        // Collect the set of affected predicates from deltas.
        let changed_predicates: std::collections::HashSet<&str> =
            deltas.iter().map(|d| d.predicate()).collect();

        let mut inner = self.inner.lock().expect("cache lock poisoned");

        let to_remove: Vec<CoreCacheKey> = inner
            .entries
            .iter()
            .filter(|(k, e)| {
                if k.dataset_id != dataset_id {
                    return false;
                }
                if e.accessed_predicates.is_empty() {
                    return true; // wildcard: invalidate always
                }
                e.accessed_predicates
                    .iter()
                    .any(|p| changed_predicates.contains(p.as_str()))
            })
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            inner.entries.remove(key);
            inner.lru.remove(key);
        }
        count
    }

    /// Return the cache hit rate in `[0, 1]`.
    ///
    /// Returns `0.0` if no requests have been made yet.
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

    /// Return the number of entries currently in the cache (including potentially expired ones
    /// that have not yet been purged).
    pub fn size(&self) -> usize {
        self.inner.lock().expect("cache lock poisoned").lru.len()
    }

    /// Forcibly remove all entries from the cache.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        inner.entries.clear();
        inner.lru.order.clear();
    }

    /// Return the raw hit count.
    pub fn hit_count(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Return the raw miss count.
    pub fn miss_count(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_key(dataset: &str, query: &str) -> CoreCacheKey {
        CoreCacheKey::new(dataset, query)
    }

    fn make_rows(count: usize) -> Vec<HashMap<String, String>> {
        (0..count)
            .map(|i| {
                let mut m = HashMap::new();
                m.insert("s".to_string(), format!("subject{}", i));
                m.insert("o".to_string(), format!("object{}", i));
                m
            })
            .collect()
    }

    // --- CoreCacheKey tests ---

    #[test]
    fn test_cache_key_same_input_same_fingerprint() {
        let k1 = make_key("ds1", "SELECT * WHERE { ?s ?p ?o }");
        let k2 = make_key("ds1", "SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(k1, k2);
        assert_eq!(k1.query_fingerprint, k2.query_fingerprint);
    }

    #[test]
    fn test_cache_key_different_datasets_different_key() {
        let k1 = make_key("ds1", "SELECT * WHERE { ?s ?p ?o }");
        let k2 = make_key("ds2", "SELECT * WHERE { ?s ?p ?o }");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_queries_different_fingerprint() {
        let k1 = make_key("ds", "SELECT ?s WHERE { ?s ?p ?o }");
        let k2 = make_key("ds", "SELECT ?o WHERE { ?s ?p ?o }");
        assert_ne!(k1.query_fingerprint, k2.query_fingerprint);
    }

    #[test]
    fn test_cache_key_hash_stable() {
        // The FNV-1a implementation must be deterministic across runs.
        let k = make_key("myds", "ASK { <s> <p> <o> }");
        // Run twice — fingerprint must be identical.
        let k2 = make_key("myds", "ASK { <s> <p> <o> }");
        assert_eq!(k.query_fingerprint, k2.query_fingerprint);
    }

    // --- CoreCacheEntry tests ---

    #[test]
    fn test_cache_entry_is_not_expired_initially() {
        let now = Instant::now();
        let entry = CoreCacheEntry {
            key: make_key("ds", "q"),
            result_rows: vec![],
            accessed_predicates: vec![],
            created_at: now,
            last_accessed: now,
            expires_at: now + Duration::from_secs(60),
            hit_count: 0,
        };
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_cache_entry_is_expired_past_deadline() {
        let past = Instant::now() - Duration::from_secs(1);
        let entry = CoreCacheEntry {
            key: make_key("ds", "q"),
            result_rows: vec![],
            accessed_predicates: vec![],
            created_at: past,
            last_accessed: past,
            expires_at: past,
            hit_count: 0,
        };
        assert!(entry.is_expired());
    }

    // --- CoreResultCache basic tests ---

    #[test]
    fn test_cache_miss_on_empty() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        let key = make_key("ds", "SELECT * WHERE { ?s ?p ?o }");
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn test_cache_hit_after_put() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        let key = make_key("ds", "SELECT * WHERE { ?s ?p ?o }");
        let rows = make_rows(3);
        cache.put(key.clone(), rows.clone(), vec!["http://p".to_string()]);

        let result = cache.get(&key).expect("cache hit expected");
        assert_eq!(result.len(), 3);
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 0);
    }

    #[test]
    fn test_cache_size_increases_on_put() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        assert_eq!(cache.size(), 0);
        cache.put(make_key("ds", "q1"), make_rows(1), vec![]);
        assert_eq!(cache.size(), 1);
        cache.put(make_key("ds", "q2"), make_rows(2), vec![]);
        assert_eq!(cache.size(), 2);
    }

    #[test]
    fn test_cache_hit_rate_pure_hits() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        let key = make_key("ds", "q");
        cache.put(key.clone(), make_rows(1), vec![]);
        cache.get(&key);
        cache.get(&key);
        assert!((cache.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_hit_rate_mixed() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        let key = make_key("ds", "q");
        cache.put(key.clone(), make_rows(1), vec![]);
        cache.get(&key); // hit
        cache.get(&make_key("ds", "other")); // miss
                                             // 1 hit, 1 miss → 0.5
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let cache = CoreResultCache::new(100, Duration::from_millis(50));
        let key = make_key("ds", "q");
        cache.put(key.clone(), make_rows(1), vec![]);
        assert!(cache.get(&key).is_some());

        thread::sleep(Duration::from_millis(100));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_put_with_custom_ttl_expires() {
        let cache = CoreResultCache::new(100, Duration::from_secs(300));
        let key = make_key("ds", "custom_ttl");
        // Override with very short TTL.
        cache.put_with_ttl(key.clone(), make_rows(1), vec![], Duration::from_millis(30));
        assert!(cache.get(&key).is_some());
        thread::sleep(Duration::from_millis(60));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        // Capacity 3 → inserting a 4th should evict the LRU entry.
        let cache = CoreResultCache::new(3, Duration::from_secs(60));
        cache.put(make_key("ds", "q1"), make_rows(1), vec![]);
        cache.put(make_key("ds", "q2"), make_rows(1), vec![]);
        cache.put(make_key("ds", "q3"), make_rows(1), vec![]);

        // Access q1 to make it MRU.
        cache.get(&make_key("ds", "q1"));

        // Insert q4 — q2 should be the LRU and get evicted.
        cache.put(make_key("ds", "q4"), make_rows(1), vec![]);

        assert!(cache.get(&make_key("ds", "q1")).is_some());
        assert!(cache.get(&make_key("ds", "q2")).is_none()); // evicted
        assert!(cache.get(&make_key("ds", "q3")).is_some());
        assert!(cache.get(&make_key("ds", "q4")).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(make_key("ds", "q1"), make_rows(1), vec![]);
        cache.put(make_key("ds", "q2"), make_rows(1), vec![]);
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    // --- Invalidation tests ---

    #[test]
    fn test_invalidate_dataset() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(make_key("dsA", "q1"), make_rows(1), vec!["p1".to_string()]);
        cache.put(make_key("dsA", "q2"), make_rows(1), vec!["p2".to_string()]);
        cache.put(make_key("dsB", "q3"), make_rows(1), vec!["p1".to_string()]);

        let removed = cache.invalidate_dataset("dsA");
        assert_eq!(removed, 2);
        assert!(cache.get(&make_key("dsA", "q1")).is_none());
        assert!(cache.get(&make_key("dsA", "q2")).is_none());
        // dsB entry untouched
        assert!(cache.get(&make_key("dsB", "q3")).is_some());
    }

    #[test]
    fn test_invalidate_predicate_specific() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(
            make_key("ds", "q1"),
            make_rows(1),
            vec!["http://p/age".to_string()],
        );
        cache.put(
            make_key("ds", "q2"),
            make_rows(1),
            vec!["http://p/name".to_string()],
        );
        cache.put(
            make_key("ds", "q3"),
            make_rows(1),
            vec!["http://p/age".to_string(), "http://p/name".to_string()],
        );

        let removed = cache.invalidate_predicate("ds", "http://p/age");
        // q1 and q3 both access age; q2 only accesses name.
        assert_eq!(removed, 2);
        assert!(cache.get(&make_key("ds", "q1")).is_none());
        assert!(cache.get(&make_key("ds", "q2")).is_some());
        assert!(cache.get(&make_key("ds", "q3")).is_none());
    }

    #[test]
    fn test_invalidate_predicate_wildcard_entry() {
        // An entry with no accessed_predicates is a wildcard and must be invalidated.
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(make_key("ds", "q_wildcard"), make_rows(1), vec![]); // no predicates

        let removed = cache.invalidate_predicate("ds", "http://p/anything");
        assert_eq!(removed, 1);
        assert!(cache.get(&make_key("ds", "q_wildcard")).is_none());
    }

    #[test]
    fn test_invalidate_on_delta_affects_matching_entries() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(
            make_key("ds", "q_age"),
            make_rows(1),
            vec!["http://p/age".to_string()],
        );
        cache.put(
            make_key("ds", "q_name"),
            make_rows(1),
            vec!["http://p/name".to_string()],
        );

        let deltas = vec![TripleDelta::Insert(
            "s".into(),
            "http://p/age".into(),
            "30".into(),
        )];
        let removed = cache.invalidate_on_delta("ds", &deltas);
        assert_eq!(removed, 1);
        assert!(cache.get(&make_key("ds", "q_age")).is_none());
        assert!(cache.get(&make_key("ds", "q_name")).is_some()); // unaffected
    }

    #[test]
    fn test_invalidate_on_delta_wildcard_entry() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(make_key("ds", "q_all"), make_rows(1), vec![]); // wildcard

        let deltas = vec![TripleDelta::Delete(
            "s".into(),
            "http://p/whatever".into(),
            "o".into(),
        )];
        let removed = cache.invalidate_on_delta("ds", &deltas);
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_invalidate_on_delta_empty_deltas_removes_nothing() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(make_key("ds", "q1"), make_rows(1), vec!["p".to_string()]);
        let removed = cache.invalidate_on_delta("ds", &[]);
        assert_eq!(removed, 0);
        assert!(cache.get(&make_key("ds", "q1")).is_some());
    }

    #[test]
    fn test_invalidate_on_delta_different_dataset_unaffected() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        cache.put(
            make_key("dsA", "q1"),
            make_rows(1),
            vec!["http://p/age".to_string()],
        );
        cache.put(
            make_key("dsB", "q2"),
            make_rows(1),
            vec!["http://p/age".to_string()],
        );

        let deltas = vec![TripleDelta::Insert(
            "s".into(),
            "http://p/age".into(),
            "5".into(),
        )];
        let removed = cache.invalidate_on_delta("dsA", &deltas);
        assert_eq!(removed, 1);
        // dsB's entry should be untouched.
        assert!(cache.get(&make_key("dsB", "q2")).is_some());
    }

    // --- Concurrent access ---

    #[test]
    fn test_concurrent_put_and_get() {
        let cache = Arc::new(CoreResultCache::new(200, Duration::from_secs(60)));
        let mut handles = vec![];

        for i in 0..8 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..25 {
                    let key = make_key("ds", &format!("query_{}_{}", i, j));
                    c.put(key.clone(), make_rows(2), vec![]);
                    let _ = c.get(&key);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        // After all threads finish the cache should be internally consistent.
        assert!(cache.size() <= 200);
    }

    #[test]
    fn test_put_overwrites_existing_key() {
        let cache = CoreResultCache::new(100, Duration::from_secs(60));
        let key = make_key("ds", "q");
        cache.put(key.clone(), make_rows(1), vec![]);
        cache.put(key.clone(), make_rows(5), vec![]);
        // Should return the most recently stored value.
        let result = cache.get(&key).expect("hit expected");
        assert_eq!(result.len(), 5);
        // Size should remain 1 (not 2).
        assert_eq!(cache.size(), 1);
    }
}
