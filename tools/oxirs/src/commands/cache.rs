//! Query result caching for improved performance
//!
//! Provides in-memory caching of SPARQL query results with TTL support

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached query result as JSON string
    result: String,
    /// Time when entry was created
    created_at: Instant,
    /// Time to live in seconds
    ttl: Duration,
    /// Number of times this entry was accessed
    hit_count: usize,
}

impl CacheEntry {
    fn new(result: String, ttl: Duration) -> Self {
        Self {
            result,
            created_at: Instant::now(),
            ttl,
            hit_count: 0,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    fn increment_hit_count(&mut self) {
        self.hit_count += 1;
    }
}

/// Query result cache with TTL and LRU eviction
pub struct QueryCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Default TTL for cache entries
    default_ttl: Duration,
    /// Maximum cache size (number of entries)
    max_size: usize,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

/// Cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub total_entries: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl QueryCache {
    /// Create a new query cache
    pub fn new(default_ttl_secs: u64, max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: Duration::from_secs(default_ttl_secs),
            max_size,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Generate cache key from dataset and query
    fn cache_key(dataset: &str, query: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        dataset.hash(&mut hasher);
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get cached result if available and not expired
    pub fn get(&self, dataset: &str, query: &str) -> Option<String> {
        let key = Self::cache_key(dataset, query);
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(entry) = cache.get_mut(&key) {
            if entry.is_expired() {
                cache.remove(&key);
                stats.misses += 1;
                stats.total_entries = cache.len();
                None
            } else {
                entry.increment_hit_count();
                stats.hits += 1;
                Some(entry.result.clone())
            }
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Store query result in cache
    pub fn set(&self, dataset: &str, query: &str, result: String) {
        self.set_with_ttl(dataset, query, result, self.default_ttl);
    }

    /// Store query result with custom TTL
    pub fn set_with_ttl(&self, dataset: &str, query: &str, result: String, ttl: Duration) {
        let key = Self::cache_key(dataset, query);
        let mut cache = self.cache.write().unwrap();

        // Evict expired entries first
        self.evict_expired(&mut cache);

        // Evict LRU entry if cache is full
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }

        cache.insert(key, CacheEntry::new(result, ttl));

        let mut stats = self.stats.write().unwrap();
        stats.total_entries = cache.len();
    }

    /// Evict expired entries
    fn evict_expired(&self, cache: &mut HashMap<String, CacheEntry>) {
        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        let mut stats = self.stats.write().unwrap();
        for key in expired_keys {
            cache.remove(&key);
            stats.evictions += 1;
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>) {
        if let Some((lru_key, _)) = cache.iter().min_by_key(|(_, entry)| entry.hit_count) {
            let lru_key = lru_key.clone();
            cache.remove(&lru_key);
            let mut stats = self.stats.write().unwrap();
            stats.evictions += 1;
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        let mut stats = self.stats.write().unwrap();
        stats.total_entries = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get number of entries in cache
    pub fn size(&self) -> usize {
        self.cache.read().unwrap().len()
    }
}

/// Global query cache singleton
static GLOBAL_CACHE: OnceLock<QueryCache> = OnceLock::new();

/// Get or initialize the global query cache
pub fn global_cache() -> &'static QueryCache {
    GLOBAL_CACHE.get_or_init(|| {
        // Default: 300 seconds TTL, 1000 max entries
        QueryCache::new(300, 1000)
    })
}

/// Cache management commands
pub mod commands {
    use super::*;

    /// Show cache statistics
    pub async fn stats_command() -> Result<()> {
        let cache = global_cache();
        let stats = cache.stats();

        println!("📊 Query Cache Statistics\n");
        println!("  Total Entries: {}", stats.total_entries);
        println!("  Cache Hits: {}", stats.hits);
        println!("  Cache Misses: {}", stats.misses);
        println!("  Hit Rate: {:.2}%", stats.hit_rate() * 100.0);
        println!("  Evictions: {}", stats.evictions);
        println!();

        Ok(())
    }

    /// Clear the query cache
    pub async fn clear_command() -> Result<()> {
        let cache = global_cache();
        let before_size = cache.size();
        cache.clear();

        println!("✅ Cache cleared");
        println!("   Removed {} entries", before_size);

        Ok(())
    }

    /// Configure cache settings
    pub async fn config_command(ttl: Option<u64>, max_size: Option<usize>) -> Result<()> {
        println!("📝 Cache Configuration\n");

        if let Some(ttl_secs) = ttl {
            println!("  TTL: {} seconds", ttl_secs);
        }

        if let Some(size) = max_size {
            println!("  Max Size: {} entries", size);
        }

        println!();
        println!("💡 Note: Cache configuration is applied at startup");
        println!("   Set OXIRS_CACHE_TTL and OXIRS_CACHE_SIZE environment variables");

        Ok(())
    }
}

/// Cache configuration from environment
#[derive(Debug)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl_secs: u64,
    pub max_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_secs: 300,  // 5 minutes
            max_size: 1000, // 1000 entries
        }
    }
}

impl CacheConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("OXIRS_CACHE_ENABLED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(true);

        let ttl_secs = std::env::var("OXIRS_CACHE_TTL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        let max_size = std::env::var("OXIRS_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        Self {
            enabled,
            ttl_secs,
            max_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = QueryCache::new(60, 100);
        let dataset = "test_dataset";
        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let result = r#"{"results": []}"#.to_string();

        // Test miss
        assert!(cache.get(dataset, query).is_none());

        // Test set and hit
        cache.set(dataset, query, result.clone());
        assert_eq!(cache.get(dataset, query), Some(result));

        // Verify stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_expiration() {
        let cache = QueryCache::new(1, 100); // 1 second TTL
        let dataset = "test_dataset";
        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let result = r#"{"results": []}"#.to_string();

        cache.set(dataset, query, result.clone());
        assert!(cache.get(dataset, query).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_secs(2));
        assert!(cache.get(dataset, query).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = QueryCache::new(60, 2); // Max 2 entries

        cache.set("ds1", "q1", "r1".to_string());
        cache.set("ds1", "q2", "r2".to_string());
        assert_eq!(cache.size(), 2);

        // Adding third entry should evict LRU
        cache.set("ds1", "q3", "r3".to_string());
        assert_eq!(cache.size(), 2);
    }
}
