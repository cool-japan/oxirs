//! Triple pattern cache: memoize repeated triple pattern lookups using LRU eviction.
//!
//! This module caches the results of triple pattern queries (where any component
//! may be a wildcard) using an LRU cache keyed by the pattern.  It also supports
//! cache invalidation when a specific triple is written so that stale results
//! are not served.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A triple pattern where `None` represents a wildcard.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    /// Subject IRI or blank node — `None` = match any.
    pub subject: Option<String>,
    /// Predicate IRI — `None` = match any.
    pub predicate: Option<String>,
    /// Object value — `None` = match any.
    pub object: Option<String>,
}

impl TriplePattern {
    /// Create a new triple pattern.
    pub fn new(subject: Option<String>, predicate: Option<String>, object: Option<String>) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Create a fully-bound pattern (no wildcards).
    pub fn bound(s: &str, p: &str, o: &str) -> Self {
        Self {
            subject: Some(s.to_string()),
            predicate: Some(p.to_string()),
            object: Some(o.to_string()),
        }
    }

    /// Create a subject-bound pattern (S ? ?).
    pub fn for_subject(s: &str) -> Self {
        Self {
            subject: Some(s.to_string()),
            predicate: None,
            object: None,
        }
    }

    /// Create a fully wildcard pattern (? ? ?).
    pub fn wildcard() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Return `true` if this pattern could match the given concrete triple.
    ///
    /// A pattern component matches when it is `None` (wildcard) or equals the
    /// corresponding triple component.
    pub fn matches(&self, s: &str, p: &str, o: &str) -> bool {
        self.subject.as_deref().map_or(true, |ps| ps == s)
            && self.predicate.as_deref().map_or(true, |pp| pp == p)
            && self.object.as_deref().map_or(true, |po| po == o)
    }
}

/// A cache key derived from a `TriplePattern` (same structure, implements Hash+Eq
/// through the derived `TriplePattern` impls).
pub type CacheKey = TriplePattern;

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// One entry in the triple cache.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached query results: (subject, predicate, object) triples.
    pub results: Vec<(String, String, String)>,
    /// How many times this entry has been retrieved.
    pub hit_count: usize,
    /// Monotonic timestamp of last access (supplied by the caller).
    pub last_accessed: u64,
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Aggregate cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted (LRU).
    pub evictions: u64,
    /// Current number of entries in the cache.
    pub size: usize,
}

// ---------------------------------------------------------------------------
// TripleCache
// ---------------------------------------------------------------------------

/// An LRU cache for triple pattern query results.
pub struct TripleCache {
    store: HashMap<CacheKey, CacheEntry>,
    capacity: usize,
    stats: CacheStats,
}

impl TripleCache {
    /// Create a new cache with the given maximum capacity (number of entries).
    pub fn new(capacity: usize) -> Self {
        Self {
            store: HashMap::new(),
            capacity,
            stats: CacheStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Look up a cached result for `pattern`.
    ///
    /// Returns a reference to the result list on a hit, or `None` on a miss.
    /// Updates `hit_count`, `last_accessed`, and cache statistics.
    pub fn get(
        &mut self,
        pattern: &TriplePattern,
        now: u64,
    ) -> Option<&Vec<(String, String, String)>> {
        if let Some(entry) = self.store.get_mut(pattern) {
            entry.hit_count += 1;
            entry.last_accessed = now;
            self.stats.hits += 1;
            // Re-borrow immutably to return the reference.
            return Some(&self.store[pattern].results);
        }
        self.stats.misses += 1;
        None
    }

    /// Store a result for `pattern`.
    ///
    /// If the cache is at capacity, the LRU entry (smallest `last_accessed`)
    /// is evicted before inserting the new entry.
    pub fn put(
        &mut self,
        pattern: TriplePattern,
        results: Vec<(String, String, String)>,
        now: u64,
    ) {
        // Evict if at capacity (and the pattern is not already present).
        if !self.store.contains_key(&pattern) && self.store.len() >= self.capacity {
            self.evict_lru();
        }

        let entry = CacheEntry {
            results,
            hit_count: 0,
            last_accessed: now,
        };
        self.store.insert(pattern, entry);
        self.stats.size = self.store.len();
    }

    /// Invalidate all cached entries whose pattern *could* match the concrete
    /// triple (s, p, o).
    ///
    /// A cached pattern must be removed if querying it again after the
    /// insertion of (s,p,o) might yield a different result set.
    pub fn invalidate_matching(&mut self, s: &str, p: &str, o: &str) {
        self.store.retain(|pattern, _| !pattern.matches(s, p, o));
        self.stats.size = self.store.len();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a reference to the current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Return the current number of cached entries.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.store.clear();
        self.stats.size = 0;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn evict_lru(&mut self) {
        if self.store.is_empty() {
            return;
        }
        // Find the key with the smallest `last_accessed` timestamp.
        let lru_key = self
            .store
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            self.store.remove(&key);
            self.stats.evictions += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> (String, String, String) {
        (s.to_string(), p.to_string(), o.to_string())
    }

    fn pattern_sp(s: &str, p: &str) -> TriplePattern {
        TriplePattern::new(Some(s.to_string()), Some(p.to_string()), None)
    }

    // -----------------------------------------------------------------------
    // TriplePattern::matches
    // -----------------------------------------------------------------------

    #[test]
    fn test_pattern_matches_wildcard() {
        let p = TriplePattern::wildcard();
        assert!(p.matches("s", "p", "o"));
    }

    #[test]
    fn test_pattern_matches_bound_exact() {
        let p = TriplePattern::bound("s", "p", "o");
        assert!(p.matches("s", "p", "o"));
        assert!(!p.matches("s2", "p", "o"));
    }

    #[test]
    fn test_pattern_matches_subject_bound() {
        let p = TriplePattern::for_subject("Alice");
        assert!(p.matches("Alice", "knows", "Bob"));
        assert!(!p.matches("Bob", "knows", "Alice"));
    }

    #[test]
    fn test_pattern_matches_partial_wildcard() {
        let p = pattern_sp(":Alice", ":knows");
        assert!(p.matches(":Alice", ":knows", ":Bob"));
        assert!(p.matches(":Alice", ":knows", ":Carol"));
        assert!(!p.matches(":Alice", ":likes", ":Bob"));
    }

    // -----------------------------------------------------------------------
    // Basic get/put
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_miss() {
        let mut cache = TripleCache::new(10);
        let p = TriplePattern::wildcard();
        assert!(cache.get(&p, 1).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_hit_after_put() {
        let mut cache = TripleCache::new(10);
        let p = TriplePattern::for_subject("Alice");
        let results = vec![triple(":Alice", ":knows", ":Bob")];
        cache.put(p.clone(), results.clone(), 1);
        let got = cache.get(&p, 2);
        assert!(got.is_some());
        assert_eq!(got.unwrap(), &results);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_size_after_put() {
        let mut cache = TripleCache::new(10);
        assert_eq!(cache.len(), 0);
        cache.put(TriplePattern::wildcard(), vec![], 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_is_empty() {
        let cache = TripleCache::new(5);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_put_updates_existing() {
        let mut cache = TripleCache::new(5);
        let p = TriplePattern::for_subject("X");
        cache.put(p.clone(), vec![triple("X", "p", "o1")], 1);
        cache.put(p.clone(), vec![triple("X", "p", "o2")], 2);
        let got = cache.get(&p, 3).unwrap();
        assert_eq!(got.len(), 1);
        // Second put overwrites
        assert_eq!(got[0].2, "o2");
        // Only 1 distinct key
        assert_eq!(cache.len(), 1);
    }

    // -----------------------------------------------------------------------
    // LRU eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_lru_eviction_on_capacity() {
        let mut cache = TripleCache::new(3);
        let p1 = TriplePattern::for_subject("A");
        let p2 = TriplePattern::for_subject("B");
        let p3 = TriplePattern::for_subject("C");
        let p4 = TriplePattern::for_subject("D");

        cache.put(p1.clone(), vec![], 1);
        cache.put(p2.clone(), vec![], 2);
        cache.put(p3.clone(), vec![], 3);
        // p1 has the smallest timestamp → should be evicted
        cache.put(p4.clone(), vec![], 4);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
        // p1 should be gone
        assert!(cache.get(&p1, 5).is_none());
        // p4 should be present
        assert!(cache.get(&p4, 6).is_some());
    }

    #[test]
    fn test_lru_eviction_respects_access_time() {
        let mut cache = TripleCache::new(2);
        let p1 = TriplePattern::for_subject("A");
        let p2 = TriplePattern::for_subject("B");
        let p3 = TriplePattern::for_subject("C");

        cache.put(p1.clone(), vec![], 1);
        cache.put(p2.clone(), vec![], 2);
        // Access p1 to refresh its timestamp
        cache.get(&p1, 10);
        // Now insert p3 — p2 (last_accessed=2) should be evicted, not p1 (last_accessed=10)
        cache.put(p3.clone(), vec![], 3);

        assert!(cache.get(&p1, 11).is_some());
        assert!(cache.get(&p3, 12).is_some());
        // p2 was the LRU
        assert!(cache.get(&p2, 13).is_none());
    }

    // -----------------------------------------------------------------------
    // Invalidation
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidate_bound_pattern() {
        let mut cache = TripleCache::new(10);
        let p = TriplePattern::bound(":Alice", ":knows", ":Bob");
        cache.put(p.clone(), vec![triple(":Alice", ":knows", ":Bob")], 1);
        cache.invalidate_matching(":Alice", ":knows", ":Bob");
        assert!(cache.get(&p, 2).is_none());
    }

    #[test]
    fn test_invalidate_wildcard_pattern_removes_all() {
        let mut cache = TripleCache::new(10);
        cache.put(TriplePattern::for_subject("A"), vec![], 1);
        cache.put(TriplePattern::for_subject("B"), vec![], 2);
        cache.put(TriplePattern::wildcard(), vec![], 3);
        // Wildcard pattern matches any triple → removed on any invalidation
        cache.invalidate_matching("A", "p", "o");
        // wildcard should be gone; "B" may remain (only if pattern doesn't match)
        assert!(cache.get(&TriplePattern::wildcard(), 4).is_none());
    }

    #[test]
    fn test_invalidate_non_matching_pattern_not_removed() {
        let mut cache = TripleCache::new(10);
        let p = TriplePattern::for_subject("Bob");
        cache.put(p.clone(), vec![triple("Bob", "age", "30")], 1);
        // Invalidate a triple about Alice — should not affect Bob's pattern
        cache.invalidate_matching("Alice", "knows", "Carol");
        assert!(cache.get(&p, 2).is_some());
    }

    #[test]
    fn test_invalidate_updates_stats_size() {
        let mut cache = TripleCache::new(10);
        cache.put(TriplePattern::bound("s", "p", "o"), vec![], 1);
        assert_eq!(cache.stats().size, 1);
        cache.invalidate_matching("s", "p", "o");
        assert_eq!(cache.stats().size, 0);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_hits_and_misses() {
        let mut cache = TripleCache::new(10);
        let p = TriplePattern::wildcard();
        cache.get(&p, 1); // miss
        cache.put(p.clone(), vec![], 2);
        cache.get(&p, 3); // hit
        cache.get(&p, 4); // hit
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn test_stats_eviction_count() {
        let mut cache = TripleCache::new(2);
        cache.put(TriplePattern::for_subject("A"), vec![], 1);
        cache.put(TriplePattern::for_subject("B"), vec![], 2);
        cache.put(TriplePattern::for_subject("C"), vec![], 3); // evict A
        cache.put(TriplePattern::for_subject("D"), vec![], 4); // evict B
        assert_eq!(cache.stats().evictions, 2);
    }

    // -----------------------------------------------------------------------
    // Clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_empties_cache() {
        let mut cache = TripleCache::new(10);
        cache.put(TriplePattern::wildcard(), vec![], 1);
        cache.put(TriplePattern::for_subject("X"), vec![], 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn test_clear_then_reuse() {
        let mut cache = TripleCache::new(5);
        cache.put(TriplePattern::for_subject("A"), vec![], 1);
        cache.clear();
        let p = TriplePattern::for_subject("B");
        cache.put(p.clone(), vec![triple("B", "p", "o")], 2);
        assert!(cache.get(&p, 3).is_some());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_capacity_one() {
        let mut cache = TripleCache::new(1);
        let p1 = TriplePattern::for_subject("A");
        let p2 = TriplePattern::for_subject("B");
        cache.put(p1.clone(), vec![], 1);
        cache.put(p2.clone(), vec![], 2); // evicts p1
        assert!(cache.get(&p1, 3).is_none());
        assert!(cache.get(&p2, 4).is_some());
    }

    #[test]
    fn test_empty_results_cached() {
        let mut cache = TripleCache::new(5);
        let p = TriplePattern::wildcard();
        cache.put(p.clone(), vec![], 1);
        let got = cache.get(&p, 2);
        assert!(got.is_some());
        assert!(got.unwrap().is_empty());
    }
}
