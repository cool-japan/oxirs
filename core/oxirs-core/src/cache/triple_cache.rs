//! Generic LRU/TTL cache for RDF triple data, query results, and prefix lookups.
//!
//! This module provides:
//! - [`TripleCache`]: Generic LRU cache with TTL for arbitrary RDF key-value data
//! - [`QueryResultCache`]: Caches SPARQL query results keyed by query hash
//! - [`PrefixCache`]: Fast namespace prefix → IRI lookup cache
//! - [`CacheStats`]: Shared hit/miss/eviction counters
//! - [`CachePolicy`]: Eviction strategy selector

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CachePolicy
// ---------------------------------------------------------------------------

/// Eviction strategy for cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    /// Least-recently-used: evict the entry accessed least recently.
    Lru,
    /// Least-frequently-used: evict the entry accessed the fewest times.
    Lfu,
    /// First-in, first-out: evict the oldest entry by insertion order.
    Fifo,
    /// Time-to-live: evict entries whose TTL has expired; fall back to LRU on capacity.
    Ttl,
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Shared, atomically-updated cache statistics.
#[derive(Debug, Default)]
pub struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    insertions: AtomicU64,
}

impl CacheStats {
    /// Create a new zeroed stats counter.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_insertion(&self) {
        self.insertions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    pub fn insertions(&self) -> u64 {
        self.insertions.load(Ordering::Relaxed)
    }

    /// Hit rate in [0, 1].  Returns 0.0 if no lookups have been made.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            h / total
        }
    }

    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Internal entry
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct Entry<V> {
    value: V,
    created_at: Instant,
    expires_at: Option<Instant>,
    last_accessed: Instant,
    access_count: u64,
    insertion_seq: u64,
}

impl<V: Clone> Entry<V> {
    fn new(value: V, ttl: Option<Duration>, seq: u64) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            expires_at: ttl.map(|d| now + d),
            last_accessed: now,
            access_count: 0,
            insertion_seq: seq,
        }
    }

    fn is_expired(&self) -> bool {
        self.expires_at
            .map(|e| Instant::now() >= e)
            .unwrap_or(false)
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

// ---------------------------------------------------------------------------
// TripleCache
// ---------------------------------------------------------------------------

struct TripleCacheInner<K, V> {
    entries: HashMap<K, Entry<V>>,
    policy: CachePolicy,
    capacity: usize,
    ttl: Option<Duration>,
    seq_counter: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> TripleCacheInner<K, V> {
    fn new(capacity: usize, policy: CachePolicy, ttl: Option<Duration>) -> Self {
        Self {
            entries: HashMap::new(),
            policy,
            capacity,
            ttl,
            seq_counter: 0,
        }
    }

    fn get_mut(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired() {
                return None;
            }
            entry.touch();
            // SAFETY: reborrow as immutable after touch
            Some(&self.entries[key].value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, value: V, stats: &CacheStats) {
        // Purge expired entries first
        self.entries.retain(|_, e| !e.is_expired());

        // Evict if at capacity
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            if let Some(victim) = self.select_victim() {
                self.entries.remove(&victim);
                stats.record_eviction();
            }
        }

        self.seq_counter += 1;
        let entry = Entry::new(value, self.ttl, self.seq_counter);
        self.entries.insert(key, entry);
        stats.record_insertion();
    }

    fn select_victim(&self) -> Option<K> {
        match self.policy {
            CachePolicy::Lru | CachePolicy::Ttl => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone()),
            CachePolicy::Lfu => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),
            CachePolicy::Fifo => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.insertion_seq)
                .map(|(k, _)| k.clone()),
        }
    }

    fn remove(&mut self, key: &K) -> bool {
        self.entries.remove(key).is_some()
    }

    fn len(&self) -> usize {
        self.entries.iter().filter(|(_, e)| !e.is_expired()).count()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn contains_key(&self, key: &K) -> bool {
        self.entries
            .get(key)
            .map(|e| !e.is_expired())
            .unwrap_or(false)
    }

    fn keys(&self) -> Vec<K> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.is_expired())
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// Generic LRU/LFU/FIFO/TTL cache for RDF data.
///
/// Thread-safe via an internal `Mutex`.  The policy is selected at construction time.
///
/// # Type Parameters
/// - `K`: Cache key type (must implement `Eq + Hash + Clone`).
/// - `V`: Cached value type (must implement `Clone`).
pub struct TripleCache<K, V> {
    inner: Mutex<TripleCacheInner<K, V>>,
    stats: Arc<CacheStats>,
}

impl<K: Eq + Hash + Clone, V: Clone> TripleCache<K, V> {
    /// Create a new cache with the given capacity, eviction policy, and optional TTL.
    pub fn new(capacity: usize, policy: CachePolicy, ttl: Option<Duration>) -> Self {
        Self {
            inner: Mutex::new(TripleCacheInner::new(capacity, policy, ttl)),
            stats: CacheStats::new(),
        }
    }

    /// Create an LRU cache with the given capacity and no TTL.
    pub fn lru(capacity: usize) -> Self {
        Self::new(capacity, CachePolicy::Lru, None)
    }

    /// Create an LRU+TTL cache.
    pub fn lru_ttl(capacity: usize, ttl: Duration) -> Self {
        Self::new(capacity, CachePolicy::Ttl, Some(ttl))
    }

    /// Look up `key` and return a clone of the value if present and not expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock().expect("TripleCache lock poisoned");
        let result = inner.get_mut(key).cloned();
        if result.is_some() {
            self.stats.record_hit();
        } else {
            self.stats.record_miss();
        }
        result
    }

    /// Insert or replace `key` → `value`.
    pub fn put(&self, key: K, value: V) {
        let mut inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.insert(key, value, &self.stats);
    }

    /// Remove `key` from the cache.  Returns `true` if it was present.
    pub fn remove(&self, key: &K) -> bool {
        let mut inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.remove(key)
    }

    /// Returns `true` if `key` is in the cache and not expired.
    pub fn contains(&self, key: &K) -> bool {
        let inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.contains_key(key)
    }

    /// Number of live (non-expired) entries.
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.clear();
    }

    /// All live keys.
    pub fn keys(&self) -> Vec<K> {
        let inner = self.inner.lock().expect("TripleCache lock poisoned");
        inner.keys()
    }

    /// Shared statistics counter.
    pub fn stats(&self) -> Arc<CacheStats> {
        Arc::clone(&self.stats)
    }
}

// ---------------------------------------------------------------------------
// QueryResultCache
// ---------------------------------------------------------------------------

/// A row of SPARQL variable bindings.
pub type SparqlRow = HashMap<String, String>;

/// Cache entry for a SPARQL query result set.
#[derive(Debug, Clone)]
pub struct QueryCacheEntry {
    /// Hashed SPARQL query string.
    pub query_hash: u64,
    /// Cached result rows.
    pub rows: Vec<SparqlRow>,
    /// Variable names in result order.
    pub variables: Vec<String>,
    /// Predicates accessed by this query (for invalidation).
    pub accessed_predicates: Vec<String>,
    /// Insertion timestamp.
    pub created_at: Instant,
}

/// Caches SPARQL query results keyed by a (dataset_id, query_hash) pair.
///
/// Internally delegates to `TripleCache<(String, u64), QueryCacheEntry>`.
pub struct QueryResultCache {
    cache: TripleCache<(String, u64), QueryCacheEntry>,
}

impl QueryResultCache {
    /// Create a new result cache with `capacity` entries and a TTL.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: TripleCache::new(capacity, CachePolicy::Ttl, Some(ttl)),
        }
    }

    /// FNV-1a hash of the SPARQL query text.
    fn hash_query(query: &str) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut h = FNV_OFFSET;
        for b in query.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// Store a result set for `(dataset_id, query)`.
    pub fn put(
        &self,
        dataset_id: &str,
        query: &str,
        rows: Vec<SparqlRow>,
        variables: Vec<String>,
        accessed_predicates: Vec<String>,
    ) {
        let hash = Self::hash_query(query);
        let entry = QueryCacheEntry {
            query_hash: hash,
            rows,
            variables,
            accessed_predicates,
            created_at: Instant::now(),
        };
        self.cache.put((dataset_id.to_string(), hash), entry);
    }

    /// Look up a cached result for `(dataset_id, query)`.
    pub fn get(&self, dataset_id: &str, query: &str) -> Option<QueryCacheEntry> {
        let hash = Self::hash_query(query);
        self.cache.get(&(dataset_id.to_string(), hash))
    }

    /// Invalidate all entries for `dataset_id` that accessed `predicate`.
    pub fn invalidate_by_predicate(&self, dataset_id: &str, predicate: &str) -> usize {
        let keys_to_remove: Vec<_> = self
            .cache
            .keys()
            .into_iter()
            .filter(|(ds, _)| ds == dataset_id)
            .filter(|k| {
                self.cache
                    .get(k)
                    .map(|e| e.accessed_predicates.iter().any(|p| p == predicate))
                    .unwrap_or(false)
            })
            .collect();
        let count = keys_to_remove.len();
        for k in keys_to_remove {
            self.cache.remove(&k);
        }
        count
    }

    /// Invalidate all entries for `dataset_id`.
    pub fn invalidate_dataset(&self, dataset_id: &str) -> usize {
        let keys_to_remove: Vec<_> = self
            .cache
            .keys()
            .into_iter()
            .filter(|(ds, _)| ds == dataset_id)
            .collect();
        let count = keys_to_remove.len();
        for k in keys_to_remove {
            self.cache.remove(&k);
        }
        count
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Shared stats.
    pub fn stats(&self) -> Arc<CacheStats> {
        self.cache.stats()
    }
}

// ---------------------------------------------------------------------------
// PrefixCache
// ---------------------------------------------------------------------------

/// Bidirectional namespace prefix ↔ IRI lookup cache.
///
/// Stores the registered prefix→IRI mappings and provides fast lookups in
/// both directions (prefix→IRI and IRI→prefix).
#[derive(Debug, Default, Clone)]
pub struct PrefixCache {
    prefix_to_iri: HashMap<String, String>,
    iri_to_prefix: HashMap<String, String>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new prefix → IRI mapping.
    ///
    /// Replaces any existing mapping for the same prefix.
    pub fn register(&mut self, prefix: &str, iri: &str) {
        // Remove stale reverse mapping if prefix was previously mapped to a different IRI
        if let Some(old_iri) = self.prefix_to_iri.get(prefix) {
            self.iri_to_prefix.remove(old_iri.as_str());
        }
        self.prefix_to_iri
            .insert(prefix.to_string(), iri.to_string());
        self.iri_to_prefix
            .insert(iri.to_string(), prefix.to_string());
    }

    /// Look up the IRI for a given prefix.  Returns `None` if not registered.
    pub fn resolve_prefix(&self, prefix: &str) -> Option<&str> {
        self.prefix_to_iri.get(prefix).map(|s| s.as_str())
    }

    /// Look up the prefix for a given namespace IRI.  Returns `None` if not registered.
    pub fn resolve_iri(&self, iri: &str) -> Option<&str> {
        self.iri_to_prefix.get(iri).map(|s| s.as_str())
    }

    /// Expand a prefixed name (e.g. `"rdf:type"`) to a full IRI.
    ///
    /// Returns `None` if the prefix is not registered or the input has no colon.
    pub fn expand(&self, prefixed: &str) -> Option<String> {
        let colon = prefixed.find(':')?;
        let prefix = &prefixed[..colon];
        let local = &prefixed[colon + 1..];
        let namespace = self.prefix_to_iri.get(prefix)?;
        Some(format!("{namespace}{local}"))
    }

    /// Compact a full IRI to a prefixed name (e.g. `"http://www.w3.org/1999/02/22-rdf-syntax-ns#type"` → `"rdf:type"`).
    ///
    /// Tries all registered namespaces; picks the longest matching namespace.
    pub fn compact(&self, iri: &str) -> Option<String> {
        let mut best: Option<(&str, &str)> = None;
        for (namespace, prefix) in &self.iri_to_prefix {
            if iri.starts_with(namespace.as_str())
                && best.map(|(ns, _)| ns.len()).unwrap_or(0) < namespace.len()
            {
                best = Some((namespace.as_str(), prefix.as_str()));
            }
        }
        best.map(|(ns, pfx)| format!("{}:{}", pfx, &iri[ns.len()..]))
    }

    /// Remove a prefix mapping.
    pub fn remove(&mut self, prefix: &str) -> bool {
        if let Some(iri) = self.prefix_to_iri.remove(prefix) {
            self.iri_to_prefix.remove(&iri);
            true
        } else {
            false
        }
    }

    /// Number of registered prefixes.
    pub fn len(&self) -> usize {
        self.prefix_to_iri.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefix_to_iri.is_empty()
    }

    /// All registered (prefix, iri) pairs.
    pub fn entries(&self) -> Vec<(&str, &str)> {
        self.prefix_to_iri
            .iter()
            .map(|(p, i)| (p.as_str(), i.as_str()))
            .collect()
    }

    /// Register the standard RDF/RDFS/OWL/XSD prefixes.
    pub fn with_standard_prefixes(mut self) -> Self {
        self.register("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        self.register("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        self.register("owl", "http://www.w3.org/2002/07/owl#");
        self.register("xsd", "http://www.w3.org/2001/XMLSchema#");
        self.register("dc", "http://purl.org/dc/elements/1.1/");
        self.register("dcterms", "http://purl.org/dc/terms/");
        self.register("foaf", "http://xmlns.com/foaf/0.1/");
        self.register("skos", "http://www.w3.org/2004/02/skos/core#");
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ---- CachePolicy ----

    #[test]
    fn test_cache_policy_variants_are_distinct() {
        assert_ne!(CachePolicy::Lru, CachePolicy::Lfu);
        assert_ne!(CachePolicy::Lfu, CachePolicy::Fifo);
        assert_ne!(CachePolicy::Fifo, CachePolicy::Ttl);
    }

    // ---- CacheStats ----

    #[test]
    fn test_cache_stats_initial_zero() {
        let s = CacheStats::new();
        assert_eq!(s.hits(), 0);
        assert_eq!(s.misses(), 0);
        assert_eq!(s.evictions(), 0);
        assert_eq!(s.insertions(), 0);
    }

    #[test]
    fn test_cache_stats_hit_rate_empty() {
        let s = CacheStats::new();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        assert_eq!(s.hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_half() {
        let s = CacheStats::new();
        s.record_hit();
        s.record_miss();
        assert!((s.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_reset() {
        let s = CacheStats::new();
        s.record_hit();
        s.record_miss();
        s.record_eviction();
        s.reset();
        assert_eq!(s.hits(), 0);
        assert_eq!(s.misses(), 0);
        assert_eq!(s.evictions(), 0);
    }

    #[test]
    fn test_cache_stats_concurrent_updates() {
        let s = Arc::new(CacheStats::default());
        let mut handles = vec![];
        for _ in 0..4 {
            let sc = Arc::clone(&s);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    sc.record_hit();
                    sc.record_miss();
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(s.hits(), 400);
        assert_eq!(s.misses(), 400);
    }

    // ---- TripleCache (LRU) ----

    #[test]
    fn test_triple_cache_lru_empty_get_returns_none() {
        let cache: TripleCache<String, String> = TripleCache::lru(10);
        assert!(cache.get(&"missing".to_string()).is_none());
    }

    #[test]
    fn test_triple_cache_lru_put_and_get() {
        let cache = TripleCache::lru(10);
        cache.put("key1".to_string(), "val1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("val1".to_string()));
    }

    #[test]
    fn test_triple_cache_lru_overwrite() {
        let cache = TripleCache::lru(10);
        cache.put("k".to_string(), "v1".to_string());
        cache.put("k".to_string(), "v2".to_string());
        assert_eq!(cache.get(&"k".to_string()), Some("v2".to_string()));
    }

    #[test]
    fn test_triple_cache_lru_eviction_at_capacity() {
        let cache: TripleCache<usize, usize> = TripleCache::lru(3);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        // Access 1 and 2 so 3 is LRU
        let _ = cache.get(&1);
        let _ = cache.get(&2);
        // Insert 4 → should evict 3 (LRU)
        cache.put(4, 40);
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&1));
        assert!(cache.contains(&2));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_triple_cache_lru_remove() {
        let cache = TripleCache::lru(10);
        cache.put("a".to_string(), 1i32);
        assert!(cache.remove(&"a".to_string()));
        assert!(!cache.remove(&"a".to_string()));
        assert!(cache.get(&"a".to_string()).is_none());
    }

    #[test]
    fn test_triple_cache_lru_clear() {
        let cache: TripleCache<i32, i32> = TripleCache::lru(10);
        for i in 0..5 {
            cache.put(i, i * 2);
        }
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_triple_cache_lru_contains() {
        let cache: TripleCache<i32, i32> = TripleCache::lru(10);
        cache.put(42, 84);
        assert!(cache.contains(&42));
        assert!(!cache.contains(&43));
    }

    #[test]
    fn test_triple_cache_lru_len() {
        let cache: TripleCache<i32, i32> = TripleCache::lru(10);
        assert_eq!(cache.len(), 0);
        cache.put(1, 1);
        assert_eq!(cache.len(), 1);
        cache.put(2, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_triple_cache_lru_keys() {
        let cache: TripleCache<i32, i32> = TripleCache::lru(10);
        cache.put(1, 10);
        cache.put(2, 20);
        let mut keys = cache.keys();
        keys.sort();
        assert_eq!(keys, vec![1, 2]);
    }

    #[test]
    fn test_triple_cache_lru_stats_incremented() {
        let cache: TripleCache<i32, i32> = TripleCache::lru(10);
        cache.put(1, 100);
        let _ = cache.get(&1);
        let _ = cache.get(&99);
        let s = cache.stats();
        assert_eq!(s.hits(), 1);
        assert_eq!(s.misses(), 1);
        assert_eq!(s.insertions(), 1);
    }

    #[test]
    fn test_triple_cache_lfu_evicts_least_used() {
        let cache: TripleCache<i32, i32> = TripleCache::new(3, CachePolicy::Lfu, None);
        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(3, 3);
        // Access 1 many times
        for _ in 0..5 {
            let _ = cache.get(&1);
        }
        // Access 2 once
        let _ = cache.get(&2);
        // 3 has zero accesses → should be LFU victim
        cache.put(4, 4);
        // Either 3 is evicted (access_count=0) or 2 (access_count=1); 1 has most
        assert!(cache.contains(&1));
    }

    #[test]
    fn test_triple_cache_fifo_evicts_oldest() {
        let cache: TripleCache<i32, i32> = TripleCache::new(3, CachePolicy::Fifo, None);
        cache.put(1, 1); // seq 1
        cache.put(2, 2); // seq 2
        cache.put(3, 3); // seq 3
                         // Access 1 to distinguish from LRU — FIFO should still evict 1 (oldest)
        let _ = cache.get(&1);
        let _ = cache.get(&1);
        cache.put(4, 4);
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&1)); // oldest evicted
    }

    #[test]
    fn test_triple_cache_ttl_expiry() {
        let ttl = Duration::from_millis(50);
        let cache: TripleCache<i32, i32> = TripleCache::lru_ttl(10, ttl);
        cache.put(1, 100);
        assert!(cache.get(&1).is_some());
        thread::sleep(Duration::from_millis(60));
        assert!(cache.get(&1).is_none());
    }

    // ---- QueryResultCache ----

    #[test]
    fn test_query_result_cache_miss_on_empty() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        assert!(qrc.get("ds", "SELECT * WHERE { ?s ?p ?o }").is_none());
    }

    #[test]
    fn test_query_result_cache_put_and_get() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        let rows = vec![[("s".to_string(), "Alice".to_string())]
            .into_iter()
            .collect()];
        qrc.put(
            "ds1",
            "SELECT ?s WHERE {?s a :Person}",
            rows.clone(),
            vec!["s".to_string()],
            vec![],
        );
        let entry = qrc
            .get("ds1", "SELECT ?s WHERE {?s a :Person}")
            .expect("should hit");
        assert_eq!(entry.rows.len(), 1);
    }

    #[test]
    fn test_query_result_cache_different_datasets_no_collision() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let rows1 = vec![[("x".to_string(), "A".to_string())].into_iter().collect()];
        let rows2 = vec![[("x".to_string(), "B".to_string())].into_iter().collect()];
        qrc.put("ds1", q, rows1, vec![], vec![]);
        qrc.put("ds2", q, rows2, vec![], vec![]);
        let e1 = qrc.get("ds1", q).expect("hit");
        let e2 = qrc.get("ds2", q).expect("hit");
        assert_ne!(e1.rows[0]["x"], e2.rows[0]["x"],);
    }

    #[test]
    fn test_query_result_cache_invalidate_by_predicate() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        qrc.put("ds", "q1", vec![], vec![], vec!["http://p/age".to_string()]);
        qrc.put(
            "ds",
            "q2",
            vec![],
            vec![],
            vec!["http://p/name".to_string()],
        );
        let removed = qrc.invalidate_by_predicate("ds", "http://p/age");
        assert_eq!(removed, 1);
        assert!(qrc.get("ds", "q1").is_none());
        assert!(qrc.get("ds", "q2").is_some());
    }

    #[test]
    fn test_query_result_cache_invalidate_dataset() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        qrc.put("ds", "q1", vec![], vec![], vec![]);
        qrc.put("ds", "q2", vec![], vec![], vec![]);
        qrc.put("other_ds", "q1", vec![], vec![], vec![]);
        let removed = qrc.invalidate_dataset("ds");
        assert_eq!(removed, 2);
        assert!(qrc.get("other_ds", "q1").is_some());
    }

    #[test]
    fn test_query_result_cache_len() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        assert_eq!(qrc.len(), 0);
        qrc.put("ds", "q1", vec![], vec![], vec![]);
        assert_eq!(qrc.len(), 1);
    }

    #[test]
    fn test_query_result_cache_is_empty() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        assert!(qrc.is_empty());
        qrc.put("ds", "q", vec![], vec![], vec![]);
        assert!(!qrc.is_empty());
    }

    #[test]
    fn test_query_result_cache_variables_preserved() {
        let qrc = QueryResultCache::new(100, Duration::from_secs(60));
        let vars = vec!["?s".to_string(), "?p".to_string(), "?o".to_string()];
        qrc.put("ds", "q", vec![], vars.clone(), vec![]);
        let entry = qrc.get("ds", "q").expect("hit");
        assert_eq!(entry.variables, vars);
    }

    // ---- PrefixCache ----

    #[test]
    fn test_prefix_cache_empty() {
        let pc = PrefixCache::new();
        assert!(pc.is_empty());
        assert_eq!(pc.len(), 0);
    }

    #[test]
    fn test_prefix_cache_register_and_resolve_prefix() {
        let mut pc = PrefixCache::new();
        pc.register("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        assert_eq!(
            pc.resolve_prefix("rdf"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    #[test]
    fn test_prefix_cache_register_and_resolve_iri() {
        let mut pc = PrefixCache::new();
        pc.register("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        assert_eq!(
            pc.resolve_iri("http://www.w3.org/2000/01/rdf-schema#"),
            Some("rdfs")
        );
    }

    #[test]
    fn test_prefix_cache_expand() {
        let mut pc = PrefixCache::new();
        pc.register("owl", "http://www.w3.org/2002/07/owl#");
        assert_eq!(
            pc.expand("owl:Class"),
            Some("http://www.w3.org/2002/07/owl#Class".to_string())
        );
    }

    #[test]
    fn test_prefix_cache_compact() {
        let mut pc = PrefixCache::new();
        pc.register("xsd", "http://www.w3.org/2001/XMLSchema#");
        assert_eq!(
            pc.compact("http://www.w3.org/2001/XMLSchema#string"),
            Some("xsd:string".to_string())
        );
    }

    #[test]
    fn test_prefix_cache_expand_unknown_prefix_returns_none() {
        let pc = PrefixCache::new();
        assert!(pc.expand("unknown:Term").is_none());
    }

    #[test]
    fn test_prefix_cache_compact_no_match_returns_none() {
        let pc = PrefixCache::new();
        assert!(pc.compact("http://example.org/x").is_none());
    }

    #[test]
    fn test_prefix_cache_overwrite_prefix() {
        let mut pc = PrefixCache::new();
        pc.register("ex", "http://example.org/");
        pc.register("ex", "http://example.com/");
        assert_eq!(pc.resolve_prefix("ex"), Some("http://example.com/"));
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn test_prefix_cache_remove() {
        let mut pc = PrefixCache::new();
        pc.register("ex", "http://example.org/");
        assert!(pc.remove("ex"));
        assert!(pc.resolve_prefix("ex").is_none());
        assert!(!pc.remove("ex")); // already gone
    }

    #[test]
    fn test_prefix_cache_standard_prefixes() {
        let pc = PrefixCache::new().with_standard_prefixes();
        assert!(pc.resolve_prefix("rdf").is_some());
        assert!(pc.resolve_prefix("rdfs").is_some());
        assert!(pc.resolve_prefix("owl").is_some());
        assert!(pc.resolve_prefix("xsd").is_some());
        assert!(pc.resolve_prefix("foaf").is_some());
    }

    #[test]
    fn test_prefix_cache_longest_namespace_wins_on_compact() {
        let mut pc = PrefixCache::new();
        pc.register("schema", "http://schema.org/");
        pc.register("schema_person", "http://schema.org/Person/");
        // The longer namespace should be chosen
        let result = pc.compact("http://schema.org/Person/name");
        assert_eq!(result, Some("schema_person:name".to_string()));
    }

    #[test]
    fn test_prefix_cache_entries_returns_all() {
        let mut pc = PrefixCache::new();
        pc.register("a", "http://a.org/");
        pc.register("b", "http://b.org/");
        let mut entries: Vec<_> = pc.entries().into_iter().map(|(p, _)| p).collect();
        entries.sort();
        assert_eq!(entries, vec!["a", "b"]);
    }
}
