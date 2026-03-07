//! GraphQL Query Result Cache with Schema-Aware Invalidation
//!
//! Caches serialised GraphQL responses keyed by (tenant, query, variables).
//! Entries are invalidated when the RDF graphs they read from are modified.
//! The cache uses an LRU eviction policy backed by an in-process `Mutex`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Stable cache key derived from tenant ID, query text, and serialised variables.
///
/// Hashing is done with a simple FNV-1a 64-bit implementation so we avoid
/// pulling in extra dependencies.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// Identifies which tenant owns this cached response.
    pub tenant_id: String,
    /// FNV-1a hash of the normalised GraphQL query string.
    pub query_hash: u64,
    /// FNV-1a hash of the serialised variables JSON (or 0 if no variables).
    pub variables_hash: u64,
}

impl CacheKey {
    /// Build a cache key from raw inputs.
    ///
    /// `variables` should be a JSON string or `None` if the query has no
    /// variables.
    pub fn new(tenant_id: &str, query: &str, variables: Option<&str>) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            query_hash: fnv1a_hash(query),
            variables_hash: variables.map(fnv1a_hash).unwrap_or(0),
        }
    }
}

/// FNV-1a 64-bit hash of a string.
fn fnv1a_hash(s: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A single cached GraphQL response entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cache key associated with this entry.
    pub key: CacheKey,
    /// The serialised JSON response body.
    pub response_json: String,
    /// RDF named graph IRIs that were read when producing this response.
    /// Used to invalidate the entry when any of these graphs changes.
    pub accessed_graphs: Vec<String>,
    /// RDF predicates that were accessed — used for finer-grained invalidation.
    pub accessed_predicates: Vec<String>,
    /// When this entry was inserted.
    pub created_at: Instant,
    /// How long this entry is valid for.
    pub ttl: Duration,
    /// Number of times this entry has been served from cache.
    pub hit_count: u64,
}

impl CacheEntry {
    /// Returns `true` if this entry has lived longer than its TTL.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }

    /// Returns the age of this entry.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns how much TTL remains (saturating at zero for expired entries).
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.created_at.elapsed())
    }
}

/// Internal cache store protected by a `Mutex`.
struct CacheStore {
    entries: HashMap<CacheKey, CacheEntry>,
    /// Insertion-order queue used to enforce the capacity limit (LRU approximation).
    lru_order: VecDeque<CacheKey>,
    /// Maps graph IRI -> set of cache keys whose responses read from that graph.
    graph_index: HashMap<String, HashSet<CacheKey>>,
    /// Maps tenant ID -> set of cache keys for that tenant.
    tenant_index: HashMap<String, HashSet<CacheKey>>,
    max_entries: usize,
}

impl CacheStore {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            graph_index: HashMap::new(),
            tenant_index: HashMap::new(),
            max_entries,
        }
    }

    /// Insert or replace an entry, maintaining indexes and capacity.
    fn insert(&mut self, entry: CacheEntry) {
        let key = entry.key.clone();

        // Update graph index
        for graph in &entry.accessed_graphs {
            self.graph_index
                .entry(graph.clone())
                .or_default()
                .insert(key.clone());
        }

        // Update tenant index
        self.tenant_index
            .entry(key.tenant_id.clone())
            .or_default()
            .insert(key.clone());

        // Remove old LRU slot for this key if it already existed
        if self.entries.contains_key(&key) {
            self.lru_order.retain(|k| k != &key);
        }

        self.entries.insert(key.clone(), entry);
        self.lru_order.push_back(key);

        // Evict oldest entries if over capacity
        while self.entries.len() > self.max_entries {
            if let Some(oldest_key) = self.lru_order.pop_front() {
                self.remove_key(&oldest_key);
            } else {
                break;
            }
        }
    }

    /// Remove a single key from all data structures.
    fn remove_key(&mut self, key: &CacheKey) {
        if let Some(entry) = self.entries.remove(key) {
            // Clean graph index
            for graph in &entry.accessed_graphs {
                if let Some(set) = self.graph_index.get_mut(graph) {
                    set.remove(key);
                }
            }
            // Clean tenant index
            if let Some(set) = self.tenant_index.get_mut(&entry.key.tenant_id) {
                set.remove(key);
            }
        }
        self.lru_order.retain(|k| k != key);
    }

    /// Mark `key` as recently used (move to back of LRU queue).
    fn touch(&mut self, key: &CacheKey) {
        self.lru_order.retain(|k| k != key);
        self.lru_order.push_back(key.clone());
    }

    /// Remove all keys matching a predicate, returning the count removed.
    fn remove_where<F>(&mut self, predicate: F) -> usize
    where
        F: Fn(&CacheKey, &CacheEntry) -> bool,
    {
        let to_remove: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(k, v)| predicate(k, v))
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for key in to_remove {
            self.remove_key(&key);
        }
        count
    }
}

/// Thread-safe, schema-aware GraphQL query result cache.
///
/// Supports:
/// - LRU eviction once the entry count reaches `max_entries`.
/// - TTL-based expiry (checked lazily on `get` and explicitly via `evict_expired`).
/// - Bulk invalidation by graph IRI or tenant ID.
/// - Hit/miss rate statistics.
pub struct GqlQueryCache {
    store: Arc<Mutex<CacheStore>>,
    default_ttl: Duration,
    hit_count: Arc<AtomicU64>,
    miss_count: Arc<AtomicU64>,
    eviction_count: Arc<AtomicU64>,
}

impl std::fmt::Debug for GqlQueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GqlQueryCache")
            .field("default_ttl", &self.default_ttl)
            .field("hits", &self.hit_count.load(Ordering::Relaxed))
            .field("misses", &self.miss_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl GqlQueryCache {
    /// Create a new cache.
    ///
    /// - `max_entries`: Maximum number of entries before LRU eviction kicks in.
    /// - `default_ttl`: TTL used when `put` does not specify a custom TTL.
    pub fn new(max_entries: usize, default_ttl: Duration) -> Self {
        Self {
            store: Arc::new(Mutex::new(CacheStore::new(max_entries))),
            default_ttl,
            hit_count: Arc::new(AtomicU64::new(0)),
            miss_count: Arc::new(AtomicU64::new(0)),
            eviction_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Look up a cached response.
    ///
    /// Returns `Some(json)` on a cache hit (updating the hit counter and
    /// LRU order).  Returns `None` on a miss or if the entry has expired
    /// (expired entries are removed eagerly).
    pub fn get(&self, key: &CacheKey) -> Option<String> {
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        match store.entries.get(key) {
            Some(entry) if entry.is_expired() => {
                let key_clone = key.clone();
                store.remove_key(&key_clone);
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                self.eviction_count.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some(entry) => {
                let response = entry.response_json.clone();
                if let Some(e) = store.entries.get_mut(key) {
                    e.hit_count += 1;
                }
                store.touch(key);
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                Some(response)
            }
            None => {
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a response in the cache with the default TTL.
    ///
    /// - `graphs`: RDF graph IRIs that were read when generating this response.
    /// - `predicates`: RDF predicates that were read.
    ///
    /// Returns `true` if the entry was inserted, `false` if the mutex was
    /// unavailable (should not happen under normal conditions).
    pub fn put(
        &self,
        key: CacheKey,
        response: String,
        graphs: Vec<String>,
        predicates: Vec<String>,
    ) -> bool {
        self.put_with_ttl(key, response, graphs, predicates, self.default_ttl)
    }

    /// Store a response with an explicit TTL override.
    pub fn put_with_ttl(
        &self,
        key: CacheKey,
        response: String,
        graphs: Vec<String>,
        predicates: Vec<String>,
        ttl: Duration,
    ) -> bool {
        let entry = CacheEntry {
            key: key.clone(),
            response_json: response,
            accessed_graphs: graphs,
            accessed_predicates: predicates,
            created_at: Instant::now(),
            ttl,
            hit_count: 0,
        };

        match self.store.lock() {
            Ok(mut store) => {
                store.insert(entry);
                true
            }
            Err(_) => false,
        }
    }

    /// Invalidate all cache entries that read from `graph`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_graph(&self, graph: &str) -> usize {
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Collect keys from graph index first to avoid borrow issues
        let keys_to_remove: Vec<CacheKey> = store
            .graph_index
            .get(graph)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            store.remove_key(&key);
        }
        self.eviction_count
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Invalidate all cache entries belonging to `tenant_id`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_tenant(&self, tenant_id: &str) -> usize {
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let keys_to_remove: Vec<CacheKey> = store
            .tenant_index
            .get(tenant_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            store.remove_key(&key);
        }
        self.eviction_count
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&self) -> usize {
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let count = store.remove_where(|_, entry| entry.is_expired());
        self.eviction_count
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Clear all entries from the cache.
    ///
    /// Returns the number of entries removed.
    pub fn clear(&self) -> usize {
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let count = store.entries.len();
        store.entries.clear();
        store.lru_order.clear();
        store.graph_index.clear();
        store.tenant_index.clear();
        count
    }

    /// Returns the cache hit rate as a value in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no requests have been made yet.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed) as f64;
        let misses = self.miss_count.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Returns the current number of entries in the cache.
    pub fn size(&self) -> usize {
        self.store.lock().map(|s| s.entries.len()).unwrap_or(0)
    }

    /// Returns cumulative hit/miss/eviction counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hit_count.load(Ordering::Relaxed),
            misses: self.miss_count.load(Ordering::Relaxed),
            evictions: self.eviction_count.load(Ordering::Relaxed),
            current_size: self.size(),
        }
    }
}

/// Snapshot of cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cache hits since creation.
    pub hits: u64,
    /// Total number of cache misses since creation.
    pub misses: u64,
    /// Total number of entries evicted (expired + LRU + manual invalidation).
    pub evictions: u64,
    /// Current number of entries in the cache.
    pub current_size: usize,
}

impl CacheStats {
    /// Returns the hit rate in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = (self.hits + self.misses) as f64;
        if total == 0.0 {
            0.0
        } else {
            self.hits as f64 / total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tenant: &str, query: &str) -> CacheKey {
        CacheKey::new(tenant, query, None)
    }

    fn key_with_vars(tenant: &str, query: &str, vars: &str) -> CacheKey {
        CacheKey::new(tenant, query, Some(vars))
    }

    #[test]
    fn test_cache_key_equality() {
        let k1 = key("t1", "{ hello }");
        let k2 = key("t1", "{ hello }");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_tenants() {
        let k1 = key("t1", "{ hello }");
        let k2 = key("t2", "{ hello }");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_queries() {
        let k1 = key("t1", "{ hello }");
        let k2 = key("t1", "{ world }");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_with_variables() {
        let k1 = key_with_vars("t1", "{ q }", r#"{"id":1}"#);
        let k2 = key_with_vars("t1", "{ q }", r#"{"id":2}"#);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_put_and_get_hit() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        let k = key("tenant1", "{ data }");
        cache.put(
            k.clone(),
            r#"{"data":{"data":"ok"}}"#.to_string(),
            vec![],
            vec![],
        );

        let result = cache.get(&k);
        assert_eq!(result.as_deref(), Some(r#"{"data":{"data":"ok"}}"#));
    }

    #[test]
    fn test_get_miss_returns_none() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        let k = key("tenant1", "{ missing }");
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn test_expired_entry_returns_none() {
        let cache = GqlQueryCache::new(100, Duration::from_nanos(1));
        let k = key("t", "q");
        cache.put_with_ttl(
            k.clone(),
            "response".to_string(),
            vec![],
            vec![],
            Duration::from_nanos(1),
        );
        // Sleep long enough for entry to expire
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get(&k).is_none());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_invalidate_by_graph() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));

        let k1 = key("t", "q1");
        let k2 = key("t", "q2");
        let k3 = key("t", "q3");

        cache.put(
            k1.clone(),
            "r1".to_string(),
            vec!["http://ex.org/g1".to_string()],
            vec![],
        );
        cache.put(
            k2.clone(),
            "r2".to_string(),
            vec![
                "http://ex.org/g1".to_string(),
                "http://ex.org/g2".to_string(),
            ],
            vec![],
        );
        cache.put(
            k3.clone(),
            "r3".to_string(),
            vec!["http://ex.org/g2".to_string()],
            vec![],
        );

        // Invalidate g1 — should remove k1 and k2
        let removed = cache.invalidate_by_graph("http://ex.org/g1");
        assert_eq!(removed, 2);
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_none());
        // k3 should survive
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_invalidate_by_tenant() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));

        let ka = CacheKey::new("tenantA", "q1", None);
        let kb = CacheKey::new("tenantB", "q1", None);

        cache.put(ka.clone(), "ra".to_string(), vec![], vec![]);
        cache.put(kb.clone(), "rb".to_string(), vec![], vec![]);

        let removed = cache.invalidate_by_tenant("tenantA");
        assert_eq!(removed, 1);
        assert!(cache.get(&ka).is_none());
        assert!(cache.get(&kb).is_some());
    }

    #[test]
    fn test_evict_expired() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));

        // Insert one expiring entry and one live entry
        let expired_key = key("t", "expiring");
        cache.put_with_ttl(
            expired_key.clone(),
            "exp".to_string(),
            vec![],
            vec![],
            Duration::from_nanos(1),
        );

        let live_key = key("t", "live");
        cache.put(live_key.clone(), "live".to_string(), vec![], vec![]);

        std::thread::sleep(Duration::from_millis(5));

        let removed = cache.evict_expired();
        assert_eq!(removed, 1);
        assert_eq!(cache.size(), 1);
        assert!(cache.get(&live_key).is_some());
    }

    #[test]
    fn test_hit_rate_calculation() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        let k = key("t", "q");
        cache.put(k.clone(), "resp".to_string(), vec![], vec![]);

        cache.get(&k); // hit
        cache.get(&key("t", "other")); // miss

        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_size_tracking() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        assert_eq!(cache.size(), 0);

        cache.put(key("t", "q1"), "r1".to_string(), vec![], vec![]);
        assert_eq!(cache.size(), 1);

        cache.put(key("t", "q2"), "r2".to_string(), vec![], vec![]);
        assert_eq!(cache.size(), 2);

        cache.invalidate_by_tenant("t");
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_lru_eviction_on_max_capacity() {
        let cache = GqlQueryCache::new(3, Duration::from_secs(60));

        let k1 = key("t", "q1");
        let k2 = key("t", "q2");
        let k3 = key("t", "q3");
        let k4 = key("t", "q4");

        cache.put(k1.clone(), "r1".to_string(), vec![], vec![]);
        cache.put(k2.clone(), "r2".to_string(), vec![], vec![]);
        cache.put(k3.clone(), "r3".to_string(), vec![], vec![]);
        // Touch k1 to make k2 the oldest
        cache.get(&k1);

        // Inserting k4 should evict k2 (least recently used)
        cache.put(k4.clone(), "r4".to_string(), vec![], vec![]);

        assert_eq!(cache.size(), 3);
        assert!(cache.get(&k2).is_none(), "k2 should have been evicted");
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k3).is_some());
        assert!(cache.get(&k4).is_some());
    }

    #[test]
    fn test_clear() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        cache.put(key("t", "q1"), "r1".to_string(), vec![], vec![]);
        cache.put(key("t", "q2"), "r2".to_string(), vec![], vec![]);

        let removed = cache.clear();
        assert_eq!(removed, 2);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_stats_snapshot() {
        let cache = GqlQueryCache::new(100, Duration::from_secs(60));
        let k = key("t", "q");
        cache.put(k.clone(), "r".to_string(), vec![], vec![]);
        cache.get(&k); // hit
        cache.get(&key("t", "miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.current_size, 1);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_hash("hello"), fnv1a_hash("hello"));
        assert_ne!(fnv1a_hash("hello"), fnv1a_hash("world"));
    }

    #[test]
    fn test_cache_entry_is_expired() {
        let entry = CacheEntry {
            key: key("t", "q"),
            response_json: String::new(),
            accessed_graphs: vec![],
            accessed_predicates: vec![],
            created_at: Instant::now(),
            ttl: Duration::from_secs(100),
            hit_count: 0,
        };
        assert!(!entry.is_expired());
    }
}
