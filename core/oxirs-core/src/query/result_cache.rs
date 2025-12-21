//! Advanced query result cache with TTL and LRU eviction
//!
//! This module provides a production-ready caching system for SPARQL query results
//! with support for:
//! - Time-To-Live (TTL) expiration
//! - Least Recently Used (LRU) eviction
//! - Memory-aware cache management
//! - Concurrent access with minimal contention
//! - Cache statistics and monitoring
//!
//! # Example
//!
//! ```
//! use oxirs_core::query::result_cache::{QueryResultCache, CacheConfig};
//! use std::time::Duration;
//!
//! let config = CacheConfig {
//!     max_entries: 1000,
//!     max_memory_bytes: 100 * 1024 * 1024, // 100 MB
//!     default_ttl: Duration::from_secs(300), // 5 minutes
//!     enable_lru: true,
//! };
//!
//! let cache = QueryResultCache::new(config);
//!
//! // Cache a query result
//! let query = "SELECT * WHERE { ?s ?p ?o }".to_string();
//! let results = vec!["result1".to_string(), "result2".to_string()];
//! cache.put(query.clone(), results.clone());
//!
//! // Retrieve from cache
//! if let Some(cached) = cache.get(&query) {
//!     println!("Cache hit! Results: {:?}", cached);
//! }
//! ```

use scirs2_core::metrics::MetricsRegistry;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for the query result cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cache entries
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Enable LRU eviction
    pub enable_lru: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_bytes: 1024 * 1024 * 1024,  // 1 GB
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_lru: true,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    /// Cached value
    value: V,
    /// Estimated size in bytes
    size_bytes: u64,
    /// Creation timestamp
    #[allow(dead_code)]
    created_at: Instant,
    /// Expiration timestamp
    expires_at: Instant,
    /// Last accessed timestamp (for LRU)
    last_accessed: Instant,
    /// Access count
    access_count: u64,
}

impl<V> CacheEntry<V> {
    fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Query result cache with TTL and LRU eviction
///
/// Thread-safe cache implementation optimized for concurrent SPARQL query workloads.
/// Uses read-write locks with minimal contention and efficient memory management.
pub struct QueryResultCache<V: Clone> {
    /// Cache configuration
    config: CacheConfig,
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry<V>>>>,
    /// LRU queue (query keys in access order)
    lru_queue: Arc<RwLock<VecDeque<String>>>,
    /// Current memory usage
    current_memory: Arc<AtomicU64>,
    /// Cache statistics
    stats: CacheStats,
    /// Metrics registry (reserved for future monitoring features)
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,
}

/// Cache statistics for monitoring
#[derive(Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: Arc<AtomicU64>,
    /// Total cache misses
    pub misses: Arc<AtomicU64>,
    /// Total evictions (LRU)
    pub evictions: Arc<AtomicU64>,
    /// Total expirations (TTL)
    pub expirations: Arc<AtomicU64>,
    /// Total puts
    pub puts: Arc<AtomicU64>,
    /// Total invalidations
    pub invalidations: Arc<AtomicU64>,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
            expirations: Arc::new(AtomicU64::new(0)),
            puts: Arc::new(AtomicU64::new(0)),
            invalidations: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get cache hit rate (0.0 to 1.0)
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

    /// Reset all statistics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.expirations.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
    }
}

impl<V: Clone> QueryResultCache<V> {
    /// Create a new query result cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let metrics = MetricsRegistry::new();

        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            current_memory: Arc::new(AtomicU64::new(0)),
            stats: CacheStats::new(),
            metrics: Arc::new(metrics),
        }
    }

    /// Put a value in the cache with default TTL
    pub fn put(&self, key: String, value: V) {
        self.put_with_ttl(key, value, self.config.default_ttl);
    }

    /// Put a value in the cache with custom TTL
    pub fn put_with_ttl(&self, key: String, value: V, ttl: Duration) {
        let now = Instant::now();
        let size_bytes = self.estimate_size(&value);

        let entry = CacheEntry {
            value,
            size_bytes,
            created_at: now,
            expires_at: now + ttl,
            last_accessed: now,
            access_count: 0,
        };

        // Check if we need to evict entries
        self.ensure_capacity(size_bytes);

        {
            let mut entries = self.entries.write().unwrap();

            // Remove old entry if exists
            if let Some(old_entry) = entries.remove(&key) {
                self.current_memory
                    .fetch_sub(old_entry.size_bytes, Ordering::Relaxed);
            }

            // Insert new entry
            entries.insert(key.clone(), entry);
            self.current_memory.fetch_add(size_bytes, Ordering::Relaxed);
        }

        // Update LRU queue
        if self.config.enable_lru {
            let mut lru = self.lru_queue.write().unwrap();
            lru.retain(|k| k != &key); // Remove if already exists
            lru.push_back(key);
        }

        self.stats.puts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a value from the cache
    pub fn get(&self, key: &str) -> Option<V> {
        // Clean expired entries periodically
        self.clean_expired();

        let mut entries = self.entries.write().unwrap();

        if let Some(entry) = entries.get_mut(key) {
            if entry.is_expired() {
                // Entry expired
                self.current_memory
                    .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                entries.remove(key);
                self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access metadata
            entry.touch();

            // Update LRU queue
            if self.config.enable_lru {
                let mut lru = self.lru_queue.write().unwrap();
                lru.retain(|k| k != key);
                lru.push_back(key.to_string());
            }

            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Invalidate (remove) a cache entry
    pub fn invalidate(&self, key: &str) -> bool {
        let mut entries = self.entries.write().unwrap();

        if let Some(entry) = entries.remove(key) {
            self.current_memory
                .fetch_sub(entry.size_bytes, Ordering::Relaxed);

            if self.config.enable_lru {
                let mut lru = self.lru_queue.write().unwrap();
                lru.retain(|k| k != key);
            }

            self.stats.invalidations.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        entries.clear();

        if self.config.enable_lru {
            let mut lru = self.lru_queue.write().unwrap();
            lru.clear();
        }

        self.current_memory.store(0, Ordering::Relaxed);
    }

    /// Get current cache size (number of entries)
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        self.current_memory.load(Ordering::Relaxed)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clean expired entries
    fn clean_expired(&self) {
        let mut entries = self.entries.write().unwrap();
        let mut to_remove = Vec::new();

        for (key, entry) in entries.iter() {
            if entry.is_expired() {
                to_remove.push((key.clone(), entry.size_bytes));
            }
        }

        for (key, size) in to_remove {
            entries.remove(&key);
            self.current_memory.fetch_sub(size, Ordering::Relaxed);
            self.stats.expirations.fetch_add(1, Ordering::Relaxed);

            if self.config.enable_lru {
                let mut lru = self.lru_queue.write().unwrap();
                lru.retain(|k| k != &key);
            }
        }
    }

    /// Ensure capacity for new entry by evicting if necessary
    fn ensure_capacity(&self, new_entry_size: u64) {
        // Check if we need to evict based on entry count
        while self.len() >= self.config.max_entries {
            self.evict_lru();
        }

        // Check if we need to evict based on memory
        while self.memory_usage() + new_entry_size > self.config.max_memory_bytes {
            self.evict_lru();
        }
    }

    /// Evict the least recently used entry
    fn evict_lru(&self) {
        if !self.config.enable_lru {
            // If LRU is disabled, evict a random entry
            let key_to_evict = {
                let entries = self.entries.read().unwrap();
                entries.keys().next().cloned()
            };

            if let Some(key) = key_to_evict {
                let mut entries = self.entries.write().unwrap();
                if let Some(entry) = entries.remove(&key) {
                    self.current_memory
                        .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
            return;
        }

        // Evict from LRU queue
        let key_to_evict = {
            let mut lru = self.lru_queue.write().unwrap();
            lru.pop_front()
        };

        if let Some(key) = key_to_evict {
            let mut entries = self.entries.write().unwrap();
            if let Some(entry) = entries.remove(&key) {
                self.current_memory
                    .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Estimate size of a value in bytes
    ///
    /// This is a simple estimation. For more accurate sizing, consider
    /// implementing a custom trait for your value types.
    fn estimate_size(&self, _value: &V) -> u64 {
        // Conservative estimate: 1KB per entry
        // In production, you might want to implement actual size calculation
        1024
    }
}

impl<V: Clone> Default for QueryResultCache<V> {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cache_operations() {
        let cache = QueryResultCache::<String>::new(CacheConfig::default());

        // Put and get
        cache.put("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get("key1"), Some("value1".to_string()));

        // Miss
        assert_eq!(cache.get("key2"), None);

        // Stats
        assert_eq!(cache.stats().hits.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats().misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_ttl_expiration() {
        let config = CacheConfig {
            default_ttl: Duration::from_millis(100),
            ..Default::default()
        };
        let cache = QueryResultCache::<String>::new(config);

        cache.put("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get("key1"), Some("value1".to_string()));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        assert_eq!(cache.get("key1"), None);
        assert_eq!(cache.stats().expirations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            enable_lru: true,
            ..Default::default()
        };
        let cache = QueryResultCache::<String>::new(config);

        // Fill cache
        cache.put("key1".to_string(), "value1".to_string());
        cache.put("key2".to_string(), "value2".to_string());
        cache.put("key3".to_string(), "value3".to_string());

        // Access key1 to make it most recently used
        cache.get("key1");

        // Add key4, should evict key2 (least recently used)
        cache.put("key4".to_string(), "value4".to_string());

        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key2"), None); // Evicted
        assert_eq!(cache.get("key3"), Some("value3".to_string()));
        assert_eq!(cache.get("key4"), Some("value4".to_string()));
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryResultCache::<String>::new(CacheConfig::default());

        cache.put("key1".to_string(), "value1".to_string());
        assert!(cache.invalidate("key1"));
        assert_eq!(cache.get("key1"), None);
        assert!(!cache.invalidate("key1")); // Already removed
    }

    #[test]
    fn test_cache_clear() {
        let cache = QueryResultCache::<String>::new(CacheConfig::default());

        cache.put("key1".to_string(), "value1".to_string());
        cache.put("key2".to_string(), "value2".to_string());

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_hit_rate() {
        let cache = QueryResultCache::<String>::new(CacheConfig::default());

        cache.put("key1".to_string(), "value1".to_string());

        cache.get("key1"); // Hit
        cache.get("key2"); // Miss
        cache.get("key1"); // Hit

        assert_eq!(cache.stats().hit_rate(), 2.0 / 3.0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(QueryResultCache::<String>::new(CacheConfig::default()));
        let mut handles = vec![];

        // Spawn multiple threads doing puts and gets
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let key = format!("key_{}", i * 100 + j);
                    let value = format!("value_{}", i * 100 + j);
                    cache_clone.put(key.clone(), value.clone());
                    cache_clone.get(&key);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify cache is in a consistent state
        assert!(cache.len() <= 1000);
    }

    #[test]
    fn test_memory_aware_eviction() {
        let config = CacheConfig {
            max_entries: 1000,
            max_memory_bytes: 5120, // 5KB (5 entries * 1KB each)
            enable_lru: true,
            ..Default::default()
        };
        let max_memory = config.max_memory_bytes;
        let cache = QueryResultCache::<String>::new(config);

        // Add 10 entries, should trigger eviction
        for i in 0..10 {
            cache.put(format!("key{}", i), format!("value{}", i));
        }

        // Should have evicted some entries to stay under memory limit
        assert!(cache.memory_usage() <= max_memory);
        assert!(cache.stats().evictions.load(Ordering::Relaxed) > 0);
    }
}
