//! Query result caching for HNSW index
//!
//! This module provides high-performance caching of query results
//! to dramatically improve performance for repeated or similar queries.

use crate::Vector;
use blake3::Hasher;
use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cached query result with expiration
#[derive(Clone, Debug)]
struct CachedResult {
    /// Search results (URI, similarity score)
    results: Vec<(String, f32)>,
    /// Time when this result was cached
    cached_at: Instant,
    /// Number of times this result has been accessed
    hit_count: usize,
}

impl CachedResult {
    fn new(results: Vec<(String, f32)>) -> Self {
        Self {
            results,
            cached_at: Instant::now(),
            hit_count: 0,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }

    fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

/// Query cache configuration
#[derive(Debug, Clone)]
pub struct QueryCacheConfig {
    /// Maximum number of cached queries
    pub max_entries: usize,
    /// Time-to-live for cached results
    pub ttl: Duration,
    /// Enable similarity-based cache lookup (find similar queries)
    pub enable_fuzzy_matching: bool,
    /// Similarity threshold for fuzzy matching (0.0-1.0)
    pub fuzzy_threshold: f32,
    /// Enable cache statistics tracking
    pub enable_stats: bool,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl: Duration::from_secs(300), // 5 minutes
            enable_fuzzy_matching: false,
            fuzzy_threshold: 0.95,
            enable_stats: true,
        }
    }
}

/// Query cache statistics
#[derive(Debug, Clone, Default)]
pub struct QueryCacheStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub expirations: u64,
}

impl QueryCacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        }
    }
}

/// High-performance query result cache for HNSW index
pub struct QueryCache {
    /// LRU cache for query results
    cache: Arc<RwLock<LruCache<u64, CachedResult>>>,
    /// Cache configuration
    config: QueryCacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<QueryCacheStats>>,
}

impl QueryCache {
    /// Create a new query cache
    pub fn new(config: QueryCacheConfig) -> Self {
        let capacity =
            NonZeroUsize::new(config.max_entries).expect("cache max_entries must be non-zero");
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(capacity))),
            config,
            stats: Arc::new(RwLock::new(QueryCacheStats::default())),
        }
    }

    /// Generate cache key from query vector and parameters
    fn generate_key(&self, query: &Vector, k: usize) -> u64 {
        let mut hasher = Hasher::new();

        // Hash the query vector
        let query_f32 = query.as_f32();
        for &val in &query_f32 {
            hasher.update(&val.to_le_bytes());
        }

        // Hash the k parameter
        hasher.update(&k.to_le_bytes());

        // Get first 8 bytes as u64
        let hash = hasher.finalize();
        let hash_bytes = hash.as_bytes();
        u64::from_le_bytes([
            hash_bytes[0],
            hash_bytes[1],
            hash_bytes[2],
            hash_bytes[3],
            hash_bytes[4],
            hash_bytes[5],
            hash_bytes[6],
            hash_bytes[7],
        ])
    }

    /// Get cached results for a query
    pub fn get(&self, query: &Vector, k: usize) -> Option<Vec<(String, f32)>> {
        if self.config.enable_stats {
            let mut stats = self.stats.write();
            stats.total_queries += 1;
        }

        let key = self.generate_key(query, k);
        let mut cache = self.cache.write();

        if let Some(cached) = cache.get_mut(&key) {
            // Check expiration
            if cached.is_expired(self.config.ttl) {
                cache.pop(&key);
                if self.config.enable_stats {
                    let mut stats = self.stats.write();
                    stats.expirations += 1;
                    stats.cache_misses += 1;
                }
                return None;
            }

            // Record hit and return results
            cached.record_hit();
            if self.config.enable_stats {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
            }
            return Some(cached.results.clone());
        }

        if self.config.enable_stats {
            let mut stats = self.stats.write();
            stats.cache_misses += 1;
        }
        None
    }

    /// Cache query results
    pub fn put(&self, query: &Vector, k: usize, results: Vec<(String, f32)>) {
        let key = self.generate_key(query, k);
        let mut cache = self.cache.write();

        let cached_result = CachedResult::new(results);

        // Check if we're evicting an entry
        if cache.len() >= self.config.max_entries && self.config.enable_stats {
            let mut stats = self.stats.write();
            stats.evictions += 1;
        }

        cache.put(key, cached_result);
    }

    /// Clear all cached results
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> QueryCacheStats {
        self.stats.read().clone()
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = QueryCacheStats::default();
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Remove expired entries (maintenance operation)
    pub fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache.write();
        let mut expired_keys = Vec::new();

        // Find expired entries
        for (key, cached) in cache.iter() {
            if cached.is_expired(self.config.ttl) {
                expired_keys.push(*key);
            }
        }

        // Remove expired entries
        let count = expired_keys.len();
        for key in expired_keys {
            cache.pop(&key);
        }

        if self.config.enable_stats && count > 0 {
            let mut stats = self.stats.write();
            stats.expirations += count as u64;
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_cache_basic() {
        let config = QueryCacheConfig::default();
        let cache = QueryCache::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results = vec![("uri1".to_string(), 0.9), ("uri2".to_string(), 0.8)];

        // Cache miss on first access
        assert!(cache.get(&query, 5).is_none());

        // Put results in cache
        cache.put(&query, 5, results.clone());

        // Cache hit on second access
        let cached = cache.get(&query, 5).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].0, "uri1");
        assert_eq!(cached[0].1, 0.9);
    }

    #[test]
    fn test_query_cache_expiration() {
        let config = QueryCacheConfig {
            ttl: Duration::from_millis(100),
            ..Default::default()
        };
        let cache = QueryCache::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results = vec![("uri1".to_string(), 0.9)];

        cache.put(&query, 5, results);

        // Should hit immediately
        assert!(cache.get(&query, 5).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should miss after expiration
        assert!(cache.get(&query, 5).is_none());
    }

    #[test]
    fn test_query_cache_stats() {
        let config = QueryCacheConfig::default();
        let cache = QueryCache::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results = vec![("uri1".to_string(), 0.9)];

        // Miss
        cache.get(&query, 5);

        // Put and hit
        cache.put(&query, 5, results);
        cache.get(&query, 5);
        cache.get(&query, 5);

        let stats = cache.get_stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_rate(), 2.0 / 3.0);
    }

    #[test]
    fn test_query_cache_cleanup() {
        let config = QueryCacheConfig {
            ttl: Duration::from_millis(100),
            ..Default::default()
        };
        let cache = QueryCache::new(config);

        // Add multiple entries
        for i in 0..5 {
            let query = Vector::new(vec![i as f32, 0.0, 0.0]);
            let results = vec![(format!("uri{}", i), 0.9)];
            cache.put(&query, 5, results);
        }

        assert_eq!(cache.len(), 5);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Cleanup expired entries
        let expired = cache.cleanup_expired();
        assert_eq!(expired, 5);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_query_cache_different_k() {
        let config = QueryCacheConfig::default();
        let cache = QueryCache::new(config);

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results_k5 = vec![("uri1".to_string(), 0.9)];
        let results_k10 = vec![("uri1".to_string(), 0.9), ("uri2".to_string(), 0.8)];

        // Cache with k=5
        cache.put(&query, 5, results_k5);

        // Cache with k=10
        cache.put(&query, 10, results_k10);

        // Different k values should have different cache entries
        let cached_k5 = cache.get(&query, 5).unwrap();
        let cached_k10 = cache.get(&query, 10).unwrap();

        assert_eq!(cached_k5.len(), 1);
        assert_eq!(cached_k10.len(), 2);
    }
}
