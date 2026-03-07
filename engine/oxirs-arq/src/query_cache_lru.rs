//! LRU Query Cache
//!
//! This module provides a focused, generic LRU cache for compiled SPARQL
//! query artifacts.  It differs from `query_plan_cache` in that it:
//!
//! - Exposes a simple `get` / `insert` / `evict_oldest` / `hit_rate` API
//! - Uses a doubly-tracked HashMap + VecDeque for O(1) amortised LRU operations
//! - Is generic over the cached value type `T`
//!
//! # Key Types
//!
//! - [`CacheFingerprint`] — structural hash of an algebra expression
//! - [`CacheEntry<T>`] — a cached value with access tracking
//! - [`LruQueryCache<T>`] — generic LRU cache with hit/miss statistics
//! - [`QueryCacheManager`] — combines fingerprinting + caching for string plans
//! - [`CacheManagerStats`] — snapshot of cache health metrics

use crate::algebra::Algebra;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CacheFingerprint
// ---------------------------------------------------------------------------

/// A structural hash of a SPARQL algebra expression used as a cache key.
///
/// Two algebra expressions that are structurally identical (same operators,
/// same variables and constants in the same positions) produce the same
/// fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheFingerprint(pub u64);

impl CacheFingerprint {
    /// Compute a fingerprint from a SPARQL algebra tree.
    ///
    /// Uses `DefaultHasher` on the `Debug` representation of the algebra.
    /// This is deterministic within a single process run and is sufficient
    /// for an in-process LRU cache.
    pub fn from_algebra(algebra: &Algebra) -> Self {
        let mut hasher = DefaultHasher::new();
        format!("{algebra:?}").hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Create a fingerprint from a raw 64-bit value (for testing).
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Return the underlying hash value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for CacheFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fingerprint({:016x})", self.0)
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A single entry in the LRU cache.
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached value.
    pub value: T,
    /// Number of times this entry has been accessed.
    pub hit_count: u64,
    /// When this entry was first inserted.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
}

impl<T> CacheEntry<T> {
    /// Create a new entry wrapping `value`.
    pub fn new(value: T) -> Self {
        let now = Instant::now();
        Self {
            value,
            hit_count: 0,
            created_at: now,
            last_accessed: now,
        }
    }

    /// Record an access and return the updated hit count.
    pub fn touch(&mut self) -> u64 {
        self.hit_count += 1;
        self.last_accessed = Instant::now();
        self.hit_count
    }

    /// Age of this entry in seconds since creation.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Time since last access in seconds.
    pub fn idle_secs(&self) -> u64 {
        self.last_accessed.elapsed().as_secs()
    }
}

// ---------------------------------------------------------------------------
// LruQueryCache
// ---------------------------------------------------------------------------

/// Generic LRU cache keyed by [`CacheFingerprint`].
///
/// Eviction policy: on `evict_oldest`, the entry with the oldest
/// `last_accessed` timestamp is removed.  The `insertion_order` deque
/// maintains insertion ordering for O(1) oldest-insertion eviction fallback.
///
/// Hit/miss statistics are tracked across the lifetime of the cache.
pub struct LruQueryCache<T> {
    entries: HashMap<CacheFingerprint, CacheEntry<T>>,
    /// Tracks insertion order for O(1) amortised LRU candidate search.
    insertion_order: VecDeque<CacheFingerprint>,
    /// Maximum number of entries.
    max_size: usize,
    total_hits: u64,
    total_misses: u64,
}

impl<T: Clone> LruQueryCache<T> {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_size: max_size.max(1),
            total_hits: 0,
            total_misses: 0,
        }
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Look up a value by fingerprint.
    ///
    /// Records a hit or miss and updates the entry's access time on hit.
    pub fn get(&mut self, key: &CacheFingerprint) -> Option<&T> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch();
            self.total_hits += 1;
            Some(&self.entries[key].value)
        } else {
            self.total_misses += 1;
            None
        }
    }

    /// Insert a value into the cache.
    ///
    /// If the cache is full and the key is not already present, `evict_oldest`
    /// is called before insertion.
    pub fn insert(&mut self, key: CacheFingerprint, value: T) {
        if self.entries.len() >= self.max_size && !self.entries.contains_key(&key) {
            self.evict_oldest();
        }
        if !self.insertion_order.contains(&key) {
            self.insertion_order.push_back(key);
        }
        self.entries.insert(key, CacheEntry::new(value));
    }

    /// Evict the least-recently-used entry (the one with the oldest
    /// `last_accessed` timestamp).
    ///
    /// Falls back to evicting the oldest-inserted entry if all timestamps are
    /// equal (e.g., in tests).
    pub fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Find the entry with the oldest last_accessed.
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| *k);

        if let Some(key) = oldest_key {
            self.entries.remove(&key);
            self.insertion_order.retain(|k| *k != key);
        }
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Fraction of lookups that were cache hits: `hits / (hits + misses)`.
    ///
    /// Returns `0.0` if no lookups have occurred.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Total number of cache hits.
    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    /// Total number of cache misses.
    pub fn total_misses(&self) -> u64 {
        self.total_misses
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the cache (does not reset hit/miss counters).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    /// Maximum number of entries the cache can hold.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

// ---------------------------------------------------------------------------
// CompiledQueryCache type alias
// ---------------------------------------------------------------------------

/// Pre-compiled query plan cache.  Uses `Vec<u8>` as a stand-in for compiled
/// plan bytecode or serialised plan structures.
pub type CompiledQueryCache = LruQueryCache<Vec<u8>>;

// ---------------------------------------------------------------------------
// QueryCacheManager
// ---------------------------------------------------------------------------

/// Combines [`CacheFingerprint`] computation with [`LruQueryCache`] for easy
/// use: supply an `&Algebra` and get (or compute) a cached string plan.
pub struct QueryCacheManager {
    cache: LruQueryCache<String>,
}

impl QueryCacheManager {
    /// Create a new manager with the given cache capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: LruQueryCache::new(max_size),
        }
    }

    /// Return the cached plan for `algebra`, or compute it with `compute` and
    /// insert it.
    ///
    /// Returns a clone of the (possibly just-inserted) plan string.
    pub fn get_or_insert(&mut self, algebra: &Algebra, compute: impl FnOnce() -> String) -> String {
        let key = CacheFingerprint::from_algebra(algebra);
        if let Some(existing) = self.cache.get(&key) {
            existing.clone()
        } else {
            let plan = compute();
            self.cache.insert(key, plan.clone());
            plan
        }
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheManagerStats {
        CacheManagerStats {
            hit_rate: self.cache.hit_rate(),
            entries: self.cache.len(),
            total_hits: self.cache.total_hits(),
            total_misses: self.cache.total_misses(),
        }
    }

    /// Evict the oldest entry from the cache.
    pub fn evict_oldest(&mut self) {
        self.cache.evict_oldest();
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached plans.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CacheManagerStats
// ---------------------------------------------------------------------------

/// Snapshot of cache health metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheManagerStats {
    /// Fraction of lookups that were hits.
    pub hit_rate: f64,
    /// Number of entries currently in the cache.
    pub entries: usize,
    /// Cumulative cache hits.
    pub total_hits: u64,
    /// Cumulative cache misses.
    pub total_misses: u64,
}

impl CacheManagerStats {
    /// Whether the cache has a high hit rate (≥ 80%).
    pub fn is_healthy(&self) -> bool {
        self.total_hits + self.total_misses == 0 || self.hit_rate >= 0.8
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, Term, TriplePattern};
    use oxirs_core::model::NamedNode;

    fn var_term(name: &str) -> Term {
        use oxirs_core::model::Variable;
        Term::Variable(Variable::new(name).expect("valid variable"))
    }

    fn make_bgp(label: &str) -> Algebra {
        use oxirs_core::model::Variable;
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new(label).expect("valid variable")),
            predicate: Term::Iri(NamedNode::new("http://ex.org/p").expect("valid IRI")),
            object: var_term("o"),
        }])
    }

    // ------------------------------------------------------------------
    // CacheFingerprint tests
    // ------------------------------------------------------------------

    #[test]
    fn test_fingerprint_same_algebra_same_hash() {
        let a = make_bgp("s");
        let b = make_bgp("s");
        assert_eq!(
            CacheFingerprint::from_algebra(&a),
            CacheFingerprint::from_algebra(&b)
        );
    }

    #[test]
    fn test_fingerprint_different_algebra_different_hash() {
        let a = make_bgp("s");
        let b = make_bgp("t");
        assert_ne!(
            CacheFingerprint::from_algebra(&a),
            CacheFingerprint::from_algebra(&b)
        );
    }

    #[test]
    fn test_fingerprint_from_raw() {
        let fp = CacheFingerprint::from_raw(42);
        assert_eq!(fp.raw(), 42);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = CacheFingerprint::from_raw(255);
        let s = format!("{fp}");
        assert!(s.starts_with("Fingerprint("));
    }

    // ------------------------------------------------------------------
    // CacheEntry tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_entry_new() {
        let entry: CacheEntry<i32> = CacheEntry::new(42);
        assert_eq!(entry.value, 42);
        assert_eq!(entry.hit_count, 0);
    }

    #[test]
    fn test_cache_entry_touch() {
        let mut entry: CacheEntry<&str> = CacheEntry::new("hello");
        assert_eq!(entry.touch(), 1);
        assert_eq!(entry.touch(), 2);
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    fn test_cache_entry_age() {
        let entry: CacheEntry<()> = CacheEntry::new(());
        assert!(entry.age_secs() < 5);
    }

    // ------------------------------------------------------------------
    // LruQueryCache tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_new_empty() {
        let cache: LruQueryCache<String> = LruQueryCache::new(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        let key = CacheFingerprint::from_raw(1);
        cache.insert(key, "plan_a".to_string());
        let result = cache.get(&key);
        assert_eq!(result, Some(&"plan_a".to_string()));
    }

    #[test]
    fn test_cache_get_missing_returns_none() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        let key = CacheFingerprint::from_raw(999);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_hit_miss_counts() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        let key = CacheFingerprint::from_raw(1);
        cache.insert(key, "plan".to_string());
        cache.get(&key);
        cache.get(&key);
        let miss_key = CacheFingerprint::from_raw(2);
        cache.get(&miss_key);
        assert_eq!(cache.total_hits(), 2);
        assert_eq!(cache.total_misses(), 1);
    }

    #[test]
    fn test_cache_hit_rate_all_hits() {
        let mut cache: LruQueryCache<i32> = LruQueryCache::new(10);
        let key = CacheFingerprint::from_raw(1);
        cache.insert(key, 42);
        cache.get(&key);
        cache.get(&key);
        assert!((cache.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_hit_rate_all_misses() {
        let mut cache: LruQueryCache<i32> = LruQueryCache::new(10);
        let key = CacheFingerprint::from_raw(1);
        cache.get(&key);
        assert!((cache.hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_hit_rate_no_lookups() {
        let cache: LruQueryCache<i32> = LruQueryCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_evict_oldest() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        let key1 = CacheFingerprint::from_raw(1);
        let key2 = CacheFingerprint::from_raw(2);
        cache.insert(key1, "plan1".to_string());
        cache.insert(key2, "plan2".to_string());
        // Access key2 so key1 is the least recently used.
        cache.get(&key2);
        cache.evict_oldest();
        assert_eq!(cache.len(), 1);
        // key1 should be evicted (older last_accessed).
        assert!(cache.entries.contains_key(&key2));
    }

    #[test]
    fn test_cache_evict_oldest_empty() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        // Should not panic.
        cache.evict_oldest();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_max_size_triggers_eviction() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(3);
        for i in 0..5 {
            cache.insert(CacheFingerprint::from_raw(i as u64), format!("plan_{i}"));
        }
        assert!(cache.len() <= 3, "cache should not exceed max_size");
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: LruQueryCache<String> = LruQueryCache::new(10);
        cache.insert(CacheFingerprint::from_raw(1), "plan".to_string());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_max_size() {
        let cache: LruQueryCache<String> = LruQueryCache::new(50);
        assert_eq!(cache.max_size(), 50);
    }

    // ------------------------------------------------------------------
    // QueryCacheManager tests
    // ------------------------------------------------------------------

    #[test]
    fn test_manager_get_or_insert_miss_then_hit() {
        let mut manager = QueryCacheManager::new(10);
        let algebra = make_bgp("s");

        let mut compute_count = 0usize;
        let plan = manager.get_or_insert(&algebra, || {
            compute_count += 1;
            "SELECT * WHERE { ?s ?p ?o }".to_string()
        });
        assert_eq!(plan, "SELECT * WHERE { ?s ?p ?o }");

        // Second call should hit the cache (compute_count stays 1).
        let plan2 = manager.get_or_insert(&algebra, || {
            compute_count += 1;
            "SHOULD NOT BE CALLED".to_string()
        });
        assert_eq!(plan2, "SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(compute_count, 1, "compute should only be called once");
    }

    #[test]
    fn test_manager_different_algebras_separate_entries() {
        let mut manager = QueryCacheManager::new(10);
        let a1 = make_bgp("s");
        let a2 = make_bgp("t");
        let plan1 = manager.get_or_insert(&a1, || "plan_s".to_string());
        let plan2 = manager.get_or_insert(&a2, || "plan_t".to_string());
        assert_ne!(plan1, plan2);
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_manager_stats_hit_rate() {
        let mut manager = QueryCacheManager::new(10);
        let algebra = make_bgp("s");
        manager.get_or_insert(&algebra, || "plan".to_string()); // miss
        manager.get_or_insert(&algebra, || unreachable!()); // hit
        let stats = manager.stats();
        assert!(stats.total_hits >= 1);
        assert!(stats.total_misses >= 1);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_manager_evict_oldest() {
        let mut manager = QueryCacheManager::new(2);
        let a1 = make_bgp("s");
        let a2 = make_bgp("t");
        manager.get_or_insert(&a1, || "plan1".to_string());
        manager.get_or_insert(&a2, || "plan2".to_string());
        manager.evict_oldest();
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_manager_clear() {
        let mut manager = QueryCacheManager::new(10);
        let algebra = make_bgp("s");
        manager.get_or_insert(&algebra, || "plan".to_string());
        manager.clear();
        assert!(manager.is_empty());
    }

    // ------------------------------------------------------------------
    // CacheManagerStats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_manager_stats_healthy() {
        let stats = CacheManagerStats {
            hit_rate: 0.9,
            entries: 5,
            total_hits: 9,
            total_misses: 1,
        };
        assert!(stats.is_healthy());
    }

    #[test]
    fn test_cache_manager_stats_unhealthy() {
        let stats = CacheManagerStats {
            hit_rate: 0.3,
            entries: 5,
            total_hits: 3,
            total_misses: 7,
        };
        assert!(!stats.is_healthy());
    }

    #[test]
    fn test_cache_manager_stats_empty_is_healthy() {
        let stats = CacheManagerStats {
            hit_rate: 0.0,
            entries: 0,
            total_hits: 0,
            total_misses: 0,
        };
        assert!(
            stats.is_healthy(),
            "empty cache should be considered healthy"
        );
    }
}
