//! Query result caching for improved performance
//!
//! This module provides a cache for SPARQL query results to reduce redundant
//! query execution. The cache uses LRU eviction and supports cache invalidation
//! when the underlying data changes.

use crate::dictionary::Term;
use crate::error::Result;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for query result cache
#[derive(Debug, Clone)]
pub struct QueryCacheConfig {
    /// Maximum number of cached query results
    pub max_entries: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Whether to enable the cache
    pub enabled: bool,
    /// Maximum result size to cache (in triples)
    pub max_result_size: usize,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(300), // 5 minutes
            enabled: true,
            max_result_size: 10000, // Don't cache huge result sets
        }
    }
}

/// Query pattern for caching (immutable key)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryPattern {
    /// Subject (None = wildcard)
    pub subject: Option<String>,
    /// Predicate (None = wildcard)
    pub predicate: Option<String>,
    /// Object (None = wildcard)
    pub object: Option<String>,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(subject: Option<&Term>, predicate: Option<&Term>, object: Option<&Term>) -> Self {
        Self {
            subject: subject.map(|t| format!("{:?}", t)),
            predicate: predicate.map(|t| format!("{:?}", t)),
            object: object.map(|t| format!("{:?}", t)),
        }
    }

    /// Check if this pattern is cacheable
    /// We avoid caching fully wildcarded queries (*, *, *) as they're too volatile
    pub fn is_cacheable(&self) -> bool {
        self.subject.is_some() || self.predicate.is_some() || self.object.is_some()
    }
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult {
    /// The query results
    results: Vec<(Term, Term, Term)>,
    /// When this entry was cached
    cached_at: Instant,
    /// Number of times this entry was accessed
    access_count: u64,
    /// Last access time
    last_accessed: Instant,
}

impl CachedResult {
    fn new(results: Vec<(Term, Term, Term)>) -> Self {
        let now = Instant::now();
        Self {
            results,
            cached_at: now,
            access_count: 0,
            last_accessed: now,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }

    fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

/// LRU entry tracker
#[derive(Debug, Clone)]
struct LruEntry {
    pattern: QueryPattern,
    last_accessed: Instant,
}

/// Query result cache with LRU eviction
pub struct QueryCache {
    /// Configuration
    config: QueryCacheConfig,
    /// Cache storage (pattern -> results)
    cache: Arc<DashMap<QueryPattern, CachedResult>>,
    /// LRU tracking queue
    lru_queue: parking_lot::Mutex<VecDeque<LruEntry>>,
    /// Statistics
    stats: QueryCacheStats,
}

/// Query cache statistics
#[derive(Debug, Default)]
pub struct QueryCacheStats {
    /// Total cache hits
    pub hits: AtomicU64,
    /// Total cache misses
    pub misses: AtomicU64,
    /// Total evictions
    pub evictions: AtomicU64,
    /// Total invalidations
    pub invalidations: AtomicU64,
    /// Current cache size
    pub current_size: AtomicUsize,
}

impl QueryCacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;

        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed)
    }
}

impl QueryCache {
    /// Create a new query cache
    pub fn new(config: QueryCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(DashMap::new()),
            lru_queue: parking_lot::Mutex::new(VecDeque::new()),
            stats: QueryCacheStats::default(),
        }
    }

    /// Get cached results for a query pattern
    pub fn get(&self, pattern: &QueryPattern) -> Option<Vec<(Term, Term, Term)>> {
        if !self.config.enabled || !pattern.is_cacheable() {
            return None;
        }

        // Try to get from cache
        if let Some(mut entry) = self.cache.get_mut(pattern) {
            // Check if expired
            if entry.is_expired(self.config.ttl) {
                // Expired - remove it
                drop(entry);
                self.cache.remove(pattern);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.stats.current_size.fetch_sub(1, Ordering::Relaxed);
                return None;
            }

            // Valid cache hit
            entry.access();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);

            // Update LRU
            self.update_lru(pattern);

            return Some(entry.results.clone());
        }

        // Cache miss
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store results in cache
    pub fn put(&self, pattern: QueryPattern, results: Vec<(Term, Term, Term)>) -> Result<()> {
        if !self.config.enabled || !pattern.is_cacheable() {
            return Ok(());
        }

        // Don't cache huge result sets
        if results.len() > self.config.max_result_size {
            return Ok(());
        }

        // Check if we need to evict
        while self.cache.len() >= self.config.max_entries {
            self.evict_lru()?;
        }

        // Insert into cache
        let entry = CachedResult::new(results);
        self.cache.insert(pattern.clone(), entry);
        self.stats.current_size.fetch_add(1, Ordering::Relaxed);

        // Update LRU
        let mut lru = self.lru_queue.lock();
        lru.push_back(LruEntry {
            pattern,
            last_accessed: Instant::now(),
        });

        Ok(())
    }

    /// Invalidate all cached results
    ///
    /// This should be called when the underlying data changes (inserts, deletes, updates)
    pub fn invalidate_all(&self) {
        let count = self.cache.len();
        self.cache.clear();
        self.lru_queue.lock().clear();

        self.stats.current_size.store(0, Ordering::Relaxed);
        self.stats
            .invalidations
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Invalidate cache entries matching a pattern
    ///
    /// This is more selective than invalidate_all() - only invalidates entries
    /// that might be affected by changes to a specific triple pattern.
    pub fn invalidate_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) {
        let mut invalidated = 0;

        // Remove all entries that overlap with the pattern
        self.cache.retain(|pattern, _| {
            let should_keep = !self.pattern_overlaps(pattern, subject, predicate, object);
            if !should_keep {
                invalidated += 1;
            }
            should_keep
        });

        // Update stats
        self.stats
            .current_size
            .fetch_sub(invalidated, Ordering::Relaxed);
        self.stats
            .invalidations
            .fetch_add(invalidated as u64, Ordering::Relaxed);

        // Clean up LRU queue (remove invalidated entries)
        let mut lru = self.lru_queue.lock();
        lru.retain(|entry| self.cache.contains_key(&entry.pattern));
    }

    /// Check if a cached pattern overlaps with a triple pattern
    fn pattern_overlaps(
        &self,
        cached: &QueryPattern,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> bool {
        // If any position matches or either is a wildcard, they overlap
        let s_overlaps = match (&cached.subject, subject) {
            (Some(cs), Some(s)) => cs == s,
            _ => true, // Wildcard always overlaps
        };

        let p_overlaps = match (&cached.predicate, predicate) {
            (Some(cp), Some(p)) => cp == p,
            _ => true,
        };

        let o_overlaps = match (&cached.object, object) {
            (Some(co), Some(o)) => co == o,
            _ => true,
        };

        s_overlaps && p_overlaps && o_overlaps
    }

    /// Evict least recently used entry
    fn evict_lru(&self) -> Result<()> {
        let mut lru = self.lru_queue.lock();

        // Find the oldest entry
        while let Some(entry) = lru.pop_front() {
            // Try to remove it (might already be gone)
            if self.cache.remove(&entry.pattern).is_some() {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.current_size.fetch_sub(1, Ordering::Relaxed);
                return Ok(());
            }
        }

        Ok(())
    }

    /// Update LRU position for a pattern
    fn update_lru(&self, pattern: &QueryPattern) {
        let mut lru = self.lru_queue.lock();

        // Find and update the entry
        if let Some(pos) = lru.iter().position(|e| &e.pattern == pattern) {
            if let Some(mut entry) = lru.remove(pos) {
                entry.last_accessed = Instant::now();
                lru.push_back(entry);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &QueryCacheStats {
        &self.stats
    }

    /// Clear all cache entries and reset statistics
    pub fn clear(&self) {
        self.cache.clear();
        self.lru_queue.lock().clear();
        self.stats.current_size.store(0, Ordering::Relaxed);
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Term;

    fn create_test_pattern(s: Option<&str>, p: Option<&str>, o: Option<&str>) -> QueryPattern {
        QueryPattern {
            subject: s.map(String::from),
            predicate: p.map(String::from),
            object: o.map(String::from),
        }
    }

    fn create_test_results(count: usize) -> Vec<(Term, Term, Term)> {
        (0..count)
            .map(|i| {
                (
                    Term::Iri(format!("http://example.org/s{}", i)),
                    Term::Iri("http://example.org/knows".to_string()),
                    Term::Iri(format!("http://example.org/o{}", i)),
                )
            })
            .collect()
    }

    #[test]
    fn test_query_cache_creation() {
        let config = QueryCacheConfig::default();
        let cache = QueryCache::new(config);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = QueryCache::new(QueryCacheConfig::default());
        let pattern = create_test_pattern(Some("s1"), None, None);
        let results = create_test_results(5);

        cache.put(pattern.clone(), results.clone()).unwrap();

        let cached = cache.get(&pattern);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 5);
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryCache::new(QueryCacheConfig::default());
        let pattern = create_test_pattern(Some("s1"), None, None);

        let cached = cache.get(&pattern);
        assert!(cached.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_hit() {
        let cache = QueryCache::new(QueryCacheConfig::default());
        let pattern = create_test_pattern(Some("s1"), None, None);
        let results = create_test_results(5);

        cache.put(pattern.clone(), results).unwrap();
        cache.get(&pattern);

        let stats = cache.stats();
        assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_cache_expiration() {
        let config = QueryCacheConfig {
            ttl: Duration::from_millis(10),
            ..Default::default()
        };
        let cache = QueryCache::new(config);
        let pattern = create_test_pattern(Some("s1"), None, None);
        let results = create_test_results(5);

        cache.put(pattern.clone(), results).unwrap();

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        let cached = cache.get(&pattern);
        assert!(cached.is_none()); // Should be expired
    }

    #[test]
    fn test_lru_eviction() {
        let config = QueryCacheConfig {
            max_entries: 3,
            ..Default::default()
        };
        let cache = QueryCache::new(config);

        // Fill cache to capacity
        for i in 0..3 {
            let pattern = create_test_pattern(Some(&format!("s{}", i)), None, None);
            cache.put(pattern, create_test_results(1)).unwrap();
        }

        assert_eq!(cache.len(), 3);

        // Add one more - should evict oldest
        let pattern4 = create_test_pattern(Some("s4"), None, None);
        cache.put(pattern4, create_test_results(1)).unwrap();

        assert_eq!(cache.len(), 3); // Still at max

        let stats = cache.stats();
        assert_eq!(stats.evictions.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_invalidate_all() {
        let cache = QueryCache::new(QueryCacheConfig::default());

        for i in 0..5 {
            let pattern = create_test_pattern(Some(&format!("s{}", i)), None, None);
            cache.put(pattern, create_test_results(1)).unwrap();
        }

        assert_eq!(cache.len(), 5);

        cache.invalidate_all();

        assert_eq!(cache.len(), 0);
        let stats = cache.stats();
        assert_eq!(stats.invalidations.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_invalidate_pattern() {
        let cache = QueryCache::new(QueryCacheConfig::default());

        // Cache some patterns
        cache
            .put(
                create_test_pattern(Some("s1"), Some("p1"), None),
                create_test_results(1),
            )
            .unwrap();
        cache
            .put(
                create_test_pattern(Some("s2"), Some("p1"), None),
                create_test_results(1),
            )
            .unwrap();
        cache
            .put(
                create_test_pattern(Some("s3"), Some("p2"), None),
                create_test_results(1),
            )
            .unwrap();

        assert_eq!(cache.len(), 3);

        // Invalidate all patterns with p1
        cache.invalidate_pattern(None, Some("p1"), None);

        assert_eq!(cache.len(), 1); // Only s3/p2 should remain
    }

    #[test]
    fn test_max_result_size() {
        let config = QueryCacheConfig {
            max_result_size: 10,
            ..Default::default()
        };
        let cache = QueryCache::new(config);

        let pattern = create_test_pattern(Some("s1"), None, None);
        let large_results = create_test_results(100); // Exceeds max

        cache.put(pattern.clone(), large_results).unwrap();

        // Should not be cached due to size
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_pattern_is_cacheable() {
        // Fully wildcarded - not cacheable
        let pattern1 = create_test_pattern(None, None, None);
        assert!(!pattern1.is_cacheable());

        // Has at least one bound position - cacheable
        let pattern2 = create_test_pattern(Some("s1"), None, None);
        assert!(pattern2.is_cacheable());

        let pattern3 = create_test_pattern(None, Some("p1"), None);
        assert!(pattern3.is_cacheable());

        let pattern4 = create_test_pattern(None, None, Some("o1"));
        assert!(pattern4.is_cacheable());
    }

    #[test]
    fn test_cache_disabled() {
        let config = QueryCacheConfig {
            enabled: false,
            ..Default::default()
        };
        let cache = QueryCache::new(config);

        let pattern = create_test_pattern(Some("s1"), None, None);
        let results = create_test_results(5);

        cache.put(pattern.clone(), results).unwrap();

        // Should not be cached when disabled
        assert_eq!(cache.len(), 0);

        let cached = cache.get(&pattern);
        assert!(cached.is_none());
    }

    #[test]
    fn test_hit_rate_calculation() {
        let cache = QueryCache::new(QueryCacheConfig::default());
        let pattern = create_test_pattern(Some("s1"), None, None);
        let results = create_test_results(5);

        cache.put(pattern.clone(), results).unwrap();

        // 3 hits
        cache.get(&pattern);
        cache.get(&pattern);
        cache.get(&pattern);

        // 2 misses
        cache.get(&create_test_pattern(Some("s2"), None, None));
        cache.get(&create_test_pattern(Some("s3"), None, None));

        let stats = cache.stats();
        assert_eq!(stats.total_requests(), 5);
        assert_eq!(stats.hit_rate(), 0.6); // 3/5 = 0.6
    }

    #[test]
    fn test_clear() {
        let cache = QueryCache::new(QueryCacheConfig::default());

        for i in 0..5 {
            let pattern = create_test_pattern(Some(&format!("s{}", i)), None, None);
            cache.put(pattern, create_test_results(1)).unwrap();
        }

        assert_eq!(cache.len(), 5);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().current_size.load(Ordering::Relaxed), 0);
    }
}
