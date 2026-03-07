//! Constraint satisfaction cache for SHACL validation
//!
//! Caches per-node validation results and supports:
//! - TTL-based expiry
//! - Targeted invalidation by focus node or accessed triple
//! - LRU-style eviction when the cache is full
//! - Thread-safe access via `Arc<Mutex<…>>`

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A triple expressed as `"<s> <p> <o>"` (the string form used as a key).
pub type TripleKey = String;

/// Cached validation result for a single focus node against a single shape.
#[derive(Debug, Clone)]
pub struct CachedValidationResult {
    /// The focus node that was validated
    pub focus_node: String,
    /// The shape ID the node was validated against
    pub shape_id: String,
    /// Whether the node is valid according to the shape
    pub is_valid: bool,
    /// Violation messages produced (empty when valid)
    pub violation_messages: Vec<String>,
    /// Timestamp at which this entry was cached
    pub cached_at: Instant,
    /// How long this entry remains valid
    pub ttl: Duration,
    /// The set of triple keys accessed while computing this result.
    ///
    /// Used to invalidate the entry when any of these triples changes.
    pub accessed_triples: HashSet<TripleKey>,
}

impl CachedValidationResult {
    /// Create a new entry.
    pub fn new(
        focus_node: impl Into<String>,
        shape_id: impl Into<String>,
        is_valid: bool,
        violation_messages: Vec<String>,
        ttl: Duration,
    ) -> Self {
        Self {
            focus_node: focus_node.into(),
            shape_id: shape_id.into(),
            is_valid,
            violation_messages,
            cached_at: Instant::now(),
            ttl,
            accessed_triples: HashSet::new(),
        }
    }

    /// Add a triple key to the dependency set.
    pub fn add_accessed_triple(&mut self, triple: impl Into<TripleKey>) {
        self.accessed_triples.insert(triple.into());
    }

    /// Returns `true` if this entry has exceeded its TTL.
    pub fn is_stale(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }

    /// Remaining lifetime of this entry, or `Duration::ZERO` if already stale.
    pub fn remaining_ttl(&self) -> Duration {
        let elapsed = self.cached_at.elapsed();
        self.ttl.checked_sub(elapsed).unwrap_or(Duration::ZERO)
    }
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Composite cache key uniquely identifying a (focus_node, shape, shape_hash) triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidationCacheKey {
    /// The focus node IRI or blank node identifier
    pub focus_node: String,
    /// The shape ID (IRI or generated UUID)
    pub shape_id: String,
    /// A content hash of the shape definition.
    ///
    /// When the shape changes, the hash changes and old entries become
    /// permanently unreachable (they will eventually be evicted as stale).
    pub shape_hash: u64,
}

impl ValidationCacheKey {
    /// Create a new cache key.
    pub fn new(
        focus_node: impl Into<String>,
        shape_id: impl Into<String>,
        shape_hash: u64,
    ) -> Self {
        Self {
            focus_node: focus_node.into(),
            shape_id: shape_id.into(),
            shape_hash,
        }
    }

    /// Compute a simple shape hash from any `Hash`-able shape representation.
    pub fn hash_shape<T: Hash>(shape: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        shape.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Internal mutable state
// ---------------------------------------------------------------------------

struct CacheInner {
    entries: HashMap<ValidationCacheKey, CachedValidationResult>,
    /// Ordered insertion history for approximate LRU eviction.
    ///
    /// The `usize` is a monotonically increasing sequence number.
    access_order: HashMap<ValidationCacheKey, usize>,
    /// Monotonically increasing counter incremented on every cache write.
    sequence: usize,
    hit_count: u64,
    miss_count: u64,
}

impl CacheInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            access_order: HashMap::new(),
            sequence: 0,
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn record_access(&mut self, key: &ValidationCacheKey) {
        self.sequence += 1;
        self.access_order.insert(key.clone(), self.sequence);
    }

    /// Evict the least-recently-used entry to make room.
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self
            .access_order
            .iter()
            .min_by_key(|(_, &seq)| seq)
            .map(|(k, v)| (k.clone(), *v))
        {
            self.entries.remove(&lru_key);
            self.access_order.remove(&lru_key);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Thread-safe, TTL-aware validation result cache.
///
/// ## Usage
///
/// ```rust
/// use oxirs_shacl::cache::validation_cache::{ValidationCache, ValidationCacheKey, CachedValidationResult};
/// use std::time::Duration;
///
/// let cache = ValidationCache::new(1000, Duration::from_secs(300));
/// let key = ValidationCacheKey::new("http://ex/Alice", "http://ex/PersonShape", 42);
/// let entry = CachedValidationResult::new(
///     "http://ex/Alice",
///     "http://ex/PersonShape",
///     true,
///     vec![],
///     Duration::from_secs(300),
/// );
/// cache.put(key.clone(), entry);
/// assert!(cache.get(&key).is_some());
/// ```
#[derive(Clone)]
pub struct ValidationCache {
    inner: Arc<Mutex<CacheInner>>,
    max_entries: usize,
    default_ttl: Duration,
}

impl ValidationCache {
    /// Create a new cache.
    ///
    /// * `max_entries` — maximum number of entries before LRU eviction kicks in
    /// * `ttl` — default time-to-live for new entries
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner::new())),
            max_entries,
            default_ttl: ttl,
        }
    }

    /// Look up a validation result.
    ///
    /// Returns `None` if the key is not present or the entry is stale.
    /// Stale entries are removed on lookup.
    pub fn get(&self, key: &ValidationCacheKey) -> Option<CachedValidationResult> {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");

        match inner.entries.get(key) {
            None => {
                inner.miss_count += 1;
                None
            }
            Some(entry) if entry.is_stale() => {
                inner.miss_count += 1;
                let k = key.clone();
                inner.entries.remove(&k);
                inner.access_order.remove(&k);
                None
            }
            Some(_) => {
                inner.hit_count += 1;
                inner.record_access(key);
                inner.entries.get(key).cloned()
            }
        }
    }

    /// Insert or update a validation result.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted first.
    pub fn put(&self, key: ValidationCacheKey, result: CachedValidationResult) {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");

        // Evict stale entries first (cheap pass)
        if inner.entries.len() >= self.max_entries {
            // Try to remove stale entries before falling back to LRU
            let stale_keys: Vec<_> = inner
                .entries
                .iter()
                .filter(|(_, v)| v.is_stale())
                .map(|(k, _)| k.clone())
                .collect();

            for sk in stale_keys {
                inner.entries.remove(&sk);
                inner.access_order.remove(&sk);
            }

            // If still at capacity, evict LRU
            if inner.entries.len() >= self.max_entries {
                inner.evict_lru();
            }
        }

        inner.record_access(&key);
        inner.entries.insert(key, result);
    }

    /// Invalidate all cache entries whose `focus_node` matches the given string.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_node(&self, focus_node: &str) -> usize {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");

        let to_remove: Vec<_> = inner
            .entries
            .keys()
            .filter(|k| k.focus_node == focus_node)
            .cloned()
            .collect();

        let count = to_remove.len();
        for k in &to_remove {
            inner.entries.remove(k);
            inner.access_order.remove(k);
        }
        count
    }

    /// Invalidate all cache entries that accessed the given triple during validation.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_triple(&self, triple_key: &str) -> usize {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");

        let to_remove: Vec<_> = inner
            .entries
            .iter()
            .filter(|(_, v)| v.accessed_triples.contains(triple_key))
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for k in &to_remove {
            inner.entries.remove(k);
            inner.access_order.remove(k);
        }
        count
    }

    /// Remove all stale (TTL-expired) entries.
    ///
    /// Returns the number of entries removed.
    pub fn evict_stale(&self) -> usize {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");

        let stale_keys: Vec<_> = inner
            .entries
            .iter()
            .filter(|(_, v)| v.is_stale())
            .map(|(k, _)| k.clone())
            .collect();

        let count = stale_keys.len();
        for k in &stale_keys {
            inner.entries.remove(k);
            inner.access_order.remove(k);
        }
        count
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        let mut inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");
        inner.entries.clear();
        inner.access_order.clear();
        inner.hit_count = 0;
        inner.miss_count = 0;
        inner.sequence = 0;
    }

    /// Return the current number of (non-stale) entries in the cache.
    pub fn size(&self) -> usize {
        let inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");
        inner.entries.values().filter(|v| !v.is_stale()).count()
    }

    /// Total number of entries (including stale ones not yet evicted).
    pub fn raw_size(&self) -> usize {
        let inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");
        inner.entries.len()
    }

    /// Cache hit rate (`hits / (hits + misses)`), or `0.0` when no lookups have occurred.
    pub fn hit_rate(&self) -> f64 {
        let inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");
        let total = inner.hit_count + inner.miss_count;
        if total == 0 {
            0.0
        } else {
            inner.hit_count as f64 / total as f64
        }
    }

    /// Returns a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self
            .inner
            .lock()
            .expect("cache lock should not be poisoned");
        let total = inner.hit_count + inner.miss_count;
        CacheStats {
            entries: inner.entries.len(),
            hit_count: inner.hit_count,
            miss_count: inner.miss_count,
            hit_rate: if total == 0 {
                0.0
            } else {
                inner.hit_count as f64 / total as f64
            },
            max_entries: self.max_entries,
            default_ttl: self.default_ttl,
        }
    }

    /// Return the default TTL for this cache instance.
    pub fn default_ttl(&self) -> Duration {
        self.default_ttl
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate: f64,
    pub max_entries: usize,
    pub default_ttl: Duration,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_entry(focus: &str, shape: &str, valid: bool, ttl: Duration) -> CachedValidationResult {
        CachedValidationResult::new(focus, shape, valid, vec![], ttl)
    }

    fn make_key(focus: &str, shape: &str) -> ValidationCacheKey {
        ValidationCacheKey::new(focus, shape, 0)
    }

    // ---- Basic get / put -------------------------------------------------

    #[test]
    fn test_put_and_get_hit() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let key = make_key("http://ex/Alice", "http://ex/PersonShape");
        let entry = make_entry(
            "http://ex/Alice",
            "http://ex/PersonShape",
            true,
            Duration::from_secs(60),
        );

        cache.put(key.clone(), entry);
        let result = cache.get(&key);
        assert!(result.is_some());
        assert!(result.expect("entry should exist").is_valid);
    }

    #[test]
    fn test_get_miss() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let key = make_key("http://ex/Alice", "http://ex/PersonShape");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_hit_rate_tracking() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let key = make_key("http://ex/Alice", "http://ex/PersonShape");
        let entry = make_entry(
            "http://ex/Alice",
            "http://ex/PersonShape",
            true,
            Duration::from_secs(60),
        );

        cache.put(key.clone(), entry);

        // 1 hit
        let _ = cache.get(&key);
        // 1 miss
        let _ = cache.get(&make_key("http://ex/Bob", "http://ex/PersonShape"));

        let stats = cache.stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ---- TTL / staleness -------------------------------------------------

    #[test]
    fn test_stale_entry_removed_on_get() {
        let cache = ValidationCache::new(100, Duration::from_millis(10));
        let key = make_key("http://ex/Alice", "http://ex/PersonShape");
        let entry = make_entry(
            "http://ex/Alice",
            "http://ex/PersonShape",
            true,
            Duration::from_millis(10),
        );

        cache.put(key.clone(), entry);

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(20));

        let result = cache.get(&key);
        assert!(result.is_none(), "stale entry should not be returned");
    }

    #[test]
    fn test_evict_stale() {
        let cache = ValidationCache::new(100, Duration::from_millis(10));

        for i in 0..5 {
            let key = make_key(&format!("http://ex/Node{i}"), "http://ex/S");
            let entry = make_entry(
                &format!("http://ex/Node{i}"),
                "http://ex/S",
                true,
                Duration::from_millis(10),
            );
            cache.put(key, entry);
        }

        thread::sleep(Duration::from_millis(20));

        let removed = cache.evict_stale();
        assert_eq!(removed, 5);
        assert_eq!(cache.raw_size(), 0);
    }

    // ---- Invalidation ----------------------------------------------------

    #[test]
    fn test_invalidate_node() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));

        cache.put(
            make_key("http://ex/Alice", "http://ex/S1"),
            make_entry(
                "http://ex/Alice",
                "http://ex/S1",
                true,
                Duration::from_secs(60),
            ),
        );
        cache.put(
            make_key("http://ex/Alice", "http://ex/S2"),
            make_entry(
                "http://ex/Alice",
                "http://ex/S2",
                true,
                Duration::from_secs(60),
            ),
        );
        cache.put(
            make_key("http://ex/Bob", "http://ex/S1"),
            make_entry(
                "http://ex/Bob",
                "http://ex/S1",
                true,
                Duration::from_secs(60),
            ),
        );

        let removed = cache.invalidate_node("http://ex/Alice");
        assert_eq!(removed, 2);

        // Bob's entry should still be present
        let bob_key = make_key("http://ex/Bob", "http://ex/S1");
        assert!(cache.get(&bob_key).is_some());
    }

    #[test]
    fn test_invalidate_triple() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));

        let triple = "<http://ex/Alice> <http://ex/age> \"30\"";

        let mut entry_a = make_entry(
            "http://ex/Alice",
            "http://ex/S1",
            true,
            Duration::from_secs(60),
        );
        entry_a.add_accessed_triple(triple);

        let entry_b = make_entry(
            "http://ex/Bob",
            "http://ex/S1",
            true,
            Duration::from_secs(60),
        );

        cache.put(make_key("http://ex/Alice", "http://ex/S1"), entry_a);
        cache.put(make_key("http://ex/Bob", "http://ex/S1"), entry_b);

        let removed = cache.invalidate_triple(triple);
        assert_eq!(removed, 1);

        // Alice's entry should be gone; Bob's should remain
        assert!(cache
            .get(&make_key("http://ex/Alice", "http://ex/S1"))
            .is_none());
        assert!(cache
            .get(&make_key("http://ex/Bob", "http://ex/S1"))
            .is_some());
    }

    // ---- Capacity / LRU eviction -----------------------------------------

    #[test]
    fn test_lru_eviction_at_capacity() {
        let cache = ValidationCache::new(3, Duration::from_secs(60));

        for i in 0..3 {
            cache.put(
                make_key(&format!("http://ex/Node{i}"), "http://ex/S"),
                make_entry(
                    &format!("http://ex/Node{i}"),
                    "http://ex/S",
                    true,
                    Duration::from_secs(60),
                ),
            );
        }

        assert_eq!(cache.raw_size(), 3);

        // This should evict the LRU entry
        cache.put(
            make_key("http://ex/Node3", "http://ex/S"),
            make_entry(
                "http://ex/Node3",
                "http://ex/S",
                true,
                Duration::from_secs(60),
            ),
        );

        assert_eq!(
            cache.raw_size(),
            3,
            "cache should remain at max capacity after LRU eviction"
        );
    }

    // ---- Concurrency -----------------------------------------------------

    #[test]
    fn test_concurrent_put_and_get() {
        let cache = Arc::new(ValidationCache::new(1000, Duration::from_secs(60)));
        let mut handles = Vec::new();

        for i in 0..10 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                let key = make_key(&format!("http://ex/Node{i}"), "http://ex/S");
                let entry = make_entry(
                    &format!("http://ex/Node{i}"),
                    "http://ex/S",
                    true,
                    Duration::from_secs(60),
                );
                c.put(key.clone(), entry);
                let r = c.get(&key);
                assert!(r.is_some(), "should find own entry");
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    // ---- Clear -----------------------------------------------------------

    #[test]
    fn test_clear() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        cache.put(
            make_key("http://ex/Alice", "http://ex/S"),
            make_entry(
                "http://ex/Alice",
                "http://ex/S",
                true,
                Duration::from_secs(60),
            ),
        );
        cache.clear();
        assert_eq!(cache.raw_size(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // ---- CachedValidationResult API --------------------------------------

    #[test]
    fn test_is_stale_false_for_fresh_entry() {
        let entry = make_entry("http://ex/A", "http://ex/S", true, Duration::from_secs(60));
        assert!(!entry.is_stale());
    }

    #[test]
    fn test_remaining_ttl_nonzero() {
        let entry = make_entry("http://ex/A", "http://ex/S", true, Duration::from_secs(60));
        assert!(entry.remaining_ttl() > Duration::ZERO);
    }

    #[test]
    fn test_shape_hash_helper() {
        let h1 = ValidationCacheKey::hash_shape(&"MyShape".to_string());
        let h2 = ValidationCacheKey::hash_shape(&"MyShape".to_string());
        assert_eq!(h1, h2);

        let h3 = ValidationCacheKey::hash_shape(&"OtherShape".to_string());
        assert_ne!(h1, h3);
    }
}

// ---------------------------------------------------------------------------
// Extended validation cache tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_cache_tests {
    use super::*;
    use std::thread;

    fn entry(focus: &str, shape: &str, valid: bool) -> CachedValidationResult {
        CachedValidationResult::new(focus, shape, valid, vec![], Duration::from_secs(60))
    }

    fn key(focus: &str, shape: &str) -> ValidationCacheKey {
        ValidationCacheKey::new(focus, shape, 0)
    }

    // ---- invalidate_node -----------------------------------------------

    #[test]
    fn test_invalidate_node_removes_single_entry() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        cache.put(
            key("http://ex/Alice", "http://ex/S"),
            entry("http://ex/Alice", "http://ex/S", true),
        );
        assert_eq!(cache.raw_size(), 1);

        let removed = cache.invalidate_node("http://ex/Alice");
        assert_eq!(removed, 1);
        assert_eq!(cache.raw_size(), 0);
    }

    #[test]
    fn test_invalidate_node_removes_multiple_shapes() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        cache.put(
            key("http://ex/Alice", "http://ex/S1"),
            entry("http://ex/Alice", "http://ex/S1", true),
        );
        cache.put(
            key("http://ex/Alice", "http://ex/S2"),
            entry("http://ex/Alice", "http://ex/S2", false),
        );
        cache.put(
            key("http://ex/Bob", "http://ex/S1"),
            entry("http://ex/Bob", "http://ex/S1", true),
        );

        let removed = cache.invalidate_node("http://ex/Alice");
        assert_eq!(removed, 2, "both Alice entries should be removed");
        assert_eq!(cache.raw_size(), 1, "Bob entry should remain");
    }

    #[test]
    fn test_invalidate_node_nonexistent_is_zero() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let removed = cache.invalidate_node("http://ex/NoSuchNode");
        assert_eq!(removed, 0);
    }

    // ---- invalidate_triple ---------------------------------------------

    #[test]
    fn test_invalidate_triple_removes_dependent_entries() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let triple_key = "http://ex/Alice/name/Bob";

        let mut e = entry("http://ex/Alice", "http://ex/S", true);
        e.add_accessed_triple(triple_key);

        cache.put(key("http://ex/Alice", "http://ex/S"), e);
        assert_eq!(cache.raw_size(), 1);

        let removed = cache.invalidate_triple(triple_key);
        assert_eq!(removed, 1);
        assert_eq!(cache.raw_size(), 0);
    }

    #[test]
    fn test_invalidate_triple_non_dependent_entry_stays() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        // Entry without any accessed triple
        cache.put(
            key("http://ex/Bob", "http://ex/S"),
            entry("http://ex/Bob", "http://ex/S", true),
        );

        let removed = cache.invalidate_triple("some:triple:key");
        assert_eq!(removed, 0);
        assert_eq!(cache.raw_size(), 1);
    }

    // ---- evict_stale ---------------------------------------------------

    #[test]
    fn test_evict_stale_removes_expired_entries() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));

        // Insert a "stale" entry with zero TTL
        let stale = CachedValidationResult::new(
            "http://ex/Alice",
            "http://ex/S",
            true,
            vec![],
            Duration::ZERO, // immediately expired
        );
        cache.put(key("http://ex/Alice", "http://ex/S"), stale);

        // Insert a fresh entry
        cache.put(
            key("http://ex/Bob", "http://ex/S"),
            entry("http://ex/Bob", "http://ex/S", true),
        );

        let evicted = cache.evict_stale();
        assert_eq!(evicted, 1, "one stale entry should be evicted");
    }

    // ---- size vs raw_size ----------------------------------------------

    #[test]
    fn test_size_excludes_stale_entries() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));

        let stale = CachedValidationResult::new(
            "http://ex/Alice",
            "http://ex/S",
            true,
            vec![],
            Duration::ZERO,
        );
        cache.put(key("http://ex/Alice", "http://ex/S"), stale);
        cache.put(
            key("http://ex/Bob", "http://ex/S"),
            entry("http://ex/Bob", "http://ex/S", true),
        );

        // raw_size counts everything including stale
        assert_eq!(cache.raw_size(), 2);
        // size() filters out stale entries
        assert!(cache.size() <= 2);
    }

    // ---- CachedValidationResult API ------------------------------------

    #[test]
    fn test_entry_is_stale_with_zero_ttl() {
        let stale = CachedValidationResult::new(
            "http://ex/Alice",
            "http://ex/S",
            true,
            vec![],
            Duration::ZERO,
        );
        assert!(stale.is_stale());
    }

    #[test]
    fn test_entry_not_stale_with_large_ttl() {
        let fresh = entry("http://ex/Alice", "http://ex/S", true);
        assert!(!fresh.is_stale());
    }

    #[test]
    fn test_remaining_ttl_is_zero_for_stale_entry() {
        let stale = CachedValidationResult::new(
            "http://ex/Alice",
            "http://ex/S",
            true,
            vec![],
            Duration::ZERO,
        );
        assert_eq!(stale.remaining_ttl(), Duration::ZERO);
    }

    #[test]
    fn test_accessed_triples_recorded() {
        let mut e = entry("http://ex/Alice", "http://ex/S", true);
        e.add_accessed_triple("triple:a");
        e.add_accessed_triple("triple:b");
        assert_eq!(e.accessed_triples.len(), 2);
    }

    // ---- ValidationCacheKey equality -----------------------------------

    #[test]
    fn test_cache_key_equality_same_inputs() {
        let k1 = ValidationCacheKey::new("http://ex/Alice", "http://ex/S", 0u64);
        let k2 = ValidationCacheKey::new("http://ex/Alice", "http://ex/S", 0u64);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_inequality_different_shape() {
        let k1 = ValidationCacheKey::new("http://ex/Alice", "http://ex/S1", 0u64);
        let k2 = ValidationCacheKey::new("http://ex/Alice", "http://ex/S2", 0u64);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_inequality_different_node() {
        let k1 = ValidationCacheKey::new("http://ex/Alice", "http://ex/S", 0u64);
        let k2 = ValidationCacheKey::new("http://ex/Bob", "http://ex/S", 0u64);
        assert_ne!(k1, k2);
    }

    // ---- CacheStats ----------------------------------------------------

    #[test]
    fn test_stats_initial_zero() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let stats = cache.stats();
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 0);
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_stats_put_count_increments() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        cache.put(
            key("http://ex/A", "http://ex/S"),
            entry("http://ex/A", "http://ex/S", true),
        );
        cache.put(
            key("http://ex/B", "http://ex/S"),
            entry("http://ex/B", "http://ex/S", true),
        );
        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
    }

    #[test]
    fn test_stats_miss_increments_on_absent_key() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let _ = cache.get(&key("http://ex/NoNode", "http://ex/S"));
        let stats = cache.stats();
        assert_eq!(stats.miss_count, 1);
    }

    // ---- default_ttl ---------------------------------------------------

    #[test]
    fn test_default_ttl_matches_constructor() {
        let ttl = Duration::from_secs(120);
        let cache = ValidationCache::new(50, ttl);
        assert_eq!(cache.default_ttl(), ttl);
    }

    // ---- Concurrent invalidation ---------------------------------------

    #[test]
    fn test_concurrent_invalidation_safety() {
        let cache = std::sync::Arc::new(ValidationCache::new(1000, Duration::from_secs(60)));

        // Pre-populate
        for i in 0..50 {
            cache.put(
                key(&format!("http://ex/Node{i}"), "http://ex/S"),
                entry(&format!("http://ex/Node{i}"), "http://ex/S", true),
            );
        }

        let cache2 = std::sync::Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for i in 0..50 {
                cache2.invalidate_node(&format!("http://ex/Node{i}"));
            }
        });

        // Simultaneously get from main thread
        for i in 0..50 {
            let _ = cache.get(&key(&format!("http://ex/Node{i}"), "http://ex/S"));
        }

        handle.join().expect("thread should not panic");
    }

    // ---- hit_rate edge cases -------------------------------------------

    #[test]
    fn test_hit_rate_zero_when_no_accesses() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_one_when_all_hits() {
        let cache = ValidationCache::new(100, Duration::from_secs(60));
        let k = key("http://ex/Alice", "http://ex/S");
        cache.put(k.clone(), entry("http://ex/Alice", "http://ex/S", true));
        let _ = cache.get(&k);
        let _ = cache.get(&k);
        // All accesses were hits
        assert!((cache.hit_rate() - 1.0).abs() < 1e-9);
    }
}
