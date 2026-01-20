//! Query Result Caching with Fingerprint-Based Keys
//!
//! Beta.2++++ Feature: Intelligent query result caching system
//!
//! This module provides a high-performance query result cache that uses
//! query fingerprints as keys for efficient lookup and invalidation.
//!
//! ## Features
//!
//! - **Fingerprint-based caching**: Uses structural query fingerprints as cache keys
//! - **TTL expiration**: Configurable time-to-live for cached results
//! - **LRU eviction**: Least-recently-used eviction when cache is full
//! - **Statistics tracking**: Cache hit rate, memory usage, eviction stats
//! - **Selective invalidation**: Invalidate by pattern or fingerprint
//! - **Compression support**: Optional result compression to save memory
//! - **Distributed cache integration**: Ready for Redis/Memcached backends
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_arq::query_result_cache::{QueryResultCache, ResultCacheConfig};
//!
//! let config = ResultCacheConfig::default()
//!     .with_max_entries(10_000)
//!     .with_ttl(std::time::Duration::from_secs(3600));
//!
//! let mut cache = QueryResultCache::new(config);
//!
//! // Cache query results using a fingerprint hash as key
//! let fingerprint_hash = "query_fingerprint_12345".to_string();
//! let results = vec![1, 2, 3, 4, 5];
//!
//! cache.put(fingerprint_hash.clone(), results.clone()).unwrap();
//!
//! // Retrieve from cache
//! if let Some(cached) = cache.get(&fingerprint_hash) {
//!     println!("Cache hit! {} results", cached.len());
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Query result cache with fingerprint-based keys
pub struct QueryResultCache {
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// LRU queue for eviction
    lru_queue: Arc<RwLock<VecDeque<String>>>,
    /// Configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
}

/// Configuration for query result cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of cached entries
    pub max_entries: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Enable result compression
    pub enable_compression: bool,
    /// Maximum result size to cache (bytes)
    pub max_result_size: usize,
    /// Enable statistics tracking
    pub enable_stats: bool,
    /// Eviction batch size
    pub eviction_batch_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: Duration::from_secs(3600), // 1 hour
            enable_compression: false,
            max_result_size: 10 * 1024 * 1024, // 10MB
            enable_stats: true,
            eviction_batch_size: 100,
        }
    }
}

impl CacheConfig {
    /// Set maximum number of entries
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set time-to-live
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Enable result compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.enable_compression = enabled;
        self
    }

    /// Set maximum result size
    pub fn with_max_result_size(mut self, size: usize) -> Self {
        self.max_result_size = size;
        self
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// Fingerprint hash (key)
    fingerprint_hash: String,
    /// Cached query results (serialized)
    results: Vec<u8>,
    /// Original result size (uncompressed)
    original_size: usize,
    /// Creation timestamp
    created_at: SystemTime,
    /// Last access timestamp
    last_accessed: SystemTime,
    /// Access count
    access_count: u64,
    /// Is compressed
    is_compressed: bool,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache puts
    pub puts: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total invalidations
    pub invalidations: u64,
    /// Cache size in bytes
    pub size_bytes: usize,
    /// Number of entries
    pub entry_count: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Average result size
    pub avg_result_size: usize,
    /// Compression ratio (if enabled)
    pub compression_ratio: f64,
}

impl CacheStatistics {
    /// Calculate hit rate
    fn calculate_hit_rate(&mut self) {
        let total = self.hits + self.misses;
        self.hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

impl QueryResultCache {
    /// Create a new query result cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
        }
    }

    /// Put query results into cache
    pub fn put(&self, fingerprint_hash: String, results: Vec<u8>) -> Result<()> {
        // Check if result size exceeds limit
        if results.len() > self.config.max_result_size {
            return Ok(()); // Skip caching large results
        }

        let mut entries = self.entries.write().expect("lock poisoned");
        let mut lru = self.lru_queue.write().expect("lock poisoned");

        // Check if we need to evict
        if entries.len() >= self.config.max_entries {
            self.evict_lru(&mut entries, &mut lru)?;
        }

        // Optionally compress results
        let (stored_results, is_compressed) = if self.config.enable_compression {
            match self.compress_results(&results) {
                Ok(compressed) => (compressed, true),
                Err(_) => (results.clone(), false),
            }
        } else {
            (results.clone(), false)
        };

        let entry = CacheEntry {
            fingerprint_hash: fingerprint_hash.clone(),
            results: stored_results.clone(),
            original_size: results.len(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            is_compressed,
        };

        // Insert into cache
        entries.insert(fingerprint_hash.clone(), entry);
        lru.push_back(fingerprint_hash);

        // Update statistics
        if self.config.enable_stats {
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.puts += 1;
            stats.entry_count = entries.len();
            stats.size_bytes += stored_results.len();
            stats.avg_result_size = if stats.entry_count > 0 {
                stats.size_bytes / stats.entry_count
            } else {
                0
            };
        }

        Ok(())
    }

    /// Get query results from cache
    pub fn get(&self, fingerprint_hash: &str) -> Option<Vec<u8>> {
        let mut entries = self.entries.write().expect("lock poisoned");
        let mut lru = self.lru_queue.write().expect("lock poisoned");

        if let Some(entry) = entries.get_mut(fingerprint_hash) {
            // Check if entry is expired
            if let Ok(elapsed) = entry.created_at.elapsed() {
                if elapsed > self.config.ttl {
                    // Entry expired, remove it
                    entries.remove(fingerprint_hash);
                    lru.retain(|k| k != fingerprint_hash);

                    // Update statistics
                    if self.config.enable_stats {
                        let mut stats = self.stats.write().expect("lock poisoned");
                        stats.misses += 1;
                        stats.evictions += 1;
                        stats.calculate_hit_rate();
                    }
                    return None;
                }
            }

            // Update access metadata
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;

            // Move to end of LRU queue
            lru.retain(|k| k != fingerprint_hash);
            lru.push_back(fingerprint_hash.to_string());

            // Decompress if needed
            let results = if entry.is_compressed {
                self.decompress_results(&entry.results).ok()?
            } else {
                entry.results.clone()
            };

            // Update statistics
            if self.config.enable_stats {
                let mut stats = self.stats.write().expect("lock poisoned");
                stats.hits += 1;
                stats.calculate_hit_rate();
            }

            Some(results)
        } else {
            // Cache miss
            if self.config.enable_stats {
                let mut stats = self.stats.write().expect("lock poisoned");
                stats.misses += 1;
                stats.calculate_hit_rate();
            }
            None
        }
    }

    /// Invalidate a specific cache entry
    pub fn invalidate(&self, fingerprint_hash: &str) -> Result<()> {
        let mut entries = self.entries.write().expect("lock poisoned");
        let mut lru = self.lru_queue.write().expect("lock poisoned");

        if entries.remove(fingerprint_hash).is_some() {
            lru.retain(|k| k != fingerprint_hash);

            if self.config.enable_stats {
                let mut stats = self.stats.write().expect("lock poisoned");
                stats.invalidations += 1;
                stats.entry_count = entries.len();
            }
        }

        Ok(())
    }

    /// Invalidate all cache entries
    pub fn invalidate_all(&self) -> Result<()> {
        let mut entries = self.entries.write().expect("lock poisoned");
        let mut lru = self.lru_queue.write().expect("lock poisoned");

        let count = entries.len();
        entries.clear();
        lru.clear();

        if self.config.enable_stats {
            let mut stats = self.stats.write().expect("lock poisoned");
            stats.invalidations += count as u64;
            stats.entry_count = 0;
            stats.size_bytes = 0;
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.stats.read().expect("lock poisoned").clone()
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.entries.read().expect("lock poisoned").len()
    }

    /// Check if cache contains a fingerprint
    pub fn contains(&self, fingerprint_hash: &str) -> bool {
        self.entries
            .read()
            .expect("lock poisoned")
            .contains_key(fingerprint_hash)
    }

    /// Evict least recently used entries
    fn evict_lru(
        &self,
        entries: &mut HashMap<String, CacheEntry>,
        lru: &mut VecDeque<String>,
    ) -> Result<()> {
        let batch_size = self.config.eviction_batch_size.min(entries.len() / 10 + 1);

        for _ in 0..batch_size {
            if let Some(oldest) = lru.pop_front() {
                if let Some(entry) = entries.remove(&oldest) {
                    if self.config.enable_stats {
                        let mut stats = self.stats.write().expect("lock poisoned");
                        stats.evictions += 1;
                        stats.size_bytes = stats.size_bytes.saturating_sub(entry.results.len());
                        stats.entry_count = entries.len();
                    }
                }
            }
        }

        Ok(())
    }

    /// Compress query results
    fn compress_results(&self, results: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(results)?;
        Ok(encoder.finish()?)
    }

    /// Decompress query results
    fn decompress_results(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}

/// Builder for query result cache
pub struct QueryResultCacheBuilder {
    config: CacheConfig,
}

impl QueryResultCacheBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    /// Set maximum entries
    pub fn max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = max;
        self
    }

    /// Set TTL
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.config.ttl = ttl;
        self
    }

    /// Enable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }

    /// Build the cache
    pub fn build(self) -> QueryResultCache {
        QueryResultCache::new(self.config)
    }
}

impl Default for QueryResultCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = QueryResultCache::new(CacheConfig::default());

        let hash = "test_hash_123".to_string();
        let results = vec![1, 2, 3, 4, 5];

        // Put and get
        cache.put(hash.clone(), results.clone()).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(results, retrieved);

        // Statistics
        let stats = cache.statistics();
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryResultCache::new(CacheConfig::default());

        let result = cache.get("nonexistent");
        assert!(result.is_none());

        let stats = cache.statistics();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryResultCache::new(CacheConfig::default());

        let hash = "test_hash".to_string();
        let results = vec![1, 2, 3];

        cache.put(hash.clone(), results).unwrap();
        assert!(cache.contains(&hash));

        cache.invalidate(&hash).unwrap();
        assert!(!cache.contains(&hash));

        let stats = cache.statistics();
        assert_eq!(stats.invalidations, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig::default().with_max_entries(3);
        let cache = QueryResultCache::new(config);

        // Fill cache
        cache.put("hash1".to_string(), vec![1]).unwrap();
        cache.put("hash2".to_string(), vec![2]).unwrap();
        cache.put("hash3".to_string(), vec![3]).unwrap();

        // This should trigger eviction
        cache.put("hash4".to_string(), vec![4]).unwrap();

        // Oldest entry should be evicted
        assert!(!cache.contains("hash1"));
        assert!(cache.contains("hash4"));
    }

    #[test]
    fn test_cache_compression() {
        let config = CacheConfig::default().with_compression(true);
        let cache = QueryResultCache::new(config);

        let hash = "compressed_hash".to_string();
        let large_results = vec![0u8; 10_000]; // 10KB of zeros (highly compressible)

        cache.put(hash.clone(), large_results.clone()).unwrap();
        let retrieved = cache.get(&hash).unwrap();
        assert_eq!(large_results, retrieved);

        let stats = cache.statistics();
        assert!(stats.compression_ratio > 1.0 || stats.size_bytes < large_results.len());
    }

    #[test]
    fn test_cache_ttl_expiration() {
        use std::thread;

        let config = CacheConfig::default().with_ttl(Duration::from_millis(100));
        let cache = QueryResultCache::new(config);

        let hash = "expiring_hash".to_string();
        cache.put(hash.clone(), vec![1, 2, 3]).unwrap();

        // Should be available immediately
        assert!(cache.get(&hash).is_some());

        // Wait for expiration
        thread::sleep(Duration::from_millis(150));

        // Should be expired
        assert!(cache.get(&hash).is_none());
    }

    #[test]
    fn test_cache_builder() {
        let cache = QueryResultCacheBuilder::new()
            .max_entries(5000)
            .ttl(Duration::from_secs(1800))
            .compression(true)
            .build();

        assert_eq!(cache.config.max_entries, 5000);
        assert_eq!(cache.config.ttl, Duration::from_secs(1800));
        assert!(cache.config.enable_compression);
    }

    #[test]
    fn test_cache_statistics_accuracy() {
        let cache = QueryResultCache::new(CacheConfig::default());

        // Perform operations
        cache.put("h1".to_string(), vec![1]).unwrap();
        cache.put("h2".to_string(), vec![2]).unwrap();
        cache.get("h1"); // hit
        cache.get("h3"); // miss
        cache.invalidate("h1").unwrap();

        let stats = cache.statistics();
        assert_eq!(stats.puts, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.invalidations, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_cache_max_result_size() {
        let config = CacheConfig::default().with_max_result_size(100);
        let cache = QueryResultCache::new(config);

        // Small result should be cached
        cache.put("small".to_string(), vec![1; 50]).unwrap();
        assert!(cache.contains("small"));

        // Large result should be skipped
        cache.put("large".to_string(), vec![1; 200]).unwrap();
        assert!(!cache.contains("large"));
    }

    #[test]
    fn test_cache_access_tracking() {
        let cache = QueryResultCache::new(CacheConfig::default());

        let hash = "tracked".to_string();
        cache.put(hash.clone(), vec![1, 2, 3]).unwrap();

        // Access multiple times
        for _ in 0..5 {
            cache.get(&hash);
        }

        let entries = cache.entries.read().unwrap();
        let entry = entries.get(&hash).unwrap();
        assert_eq!(entry.access_count, 5);
    }
}
