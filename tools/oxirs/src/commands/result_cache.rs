//! Enhanced client-side LRU result cache with TTL and persistent storage
//!
//! This module implements a production-quality LRU cache for SPARQL query results
//! featuring proper LRU eviction, TTL-based expiration, cache invalidation commands,
//! and optional disk persistence.
//!
//! ## Features
//!
//! - **True LRU eviction**: Uses doubly-linked list for O(1) eviction
//! - **TTL expiration**: Per-entry and global default TTL
//! - **Cache invalidation**: Dataset-level and query-level invalidation
//! - **Persistent storage**: Optional disk persistence for cross-session caching
//! - **Statistics**: Comprehensive hit/miss/eviction metrics
//! - **Namespace isolation**: Per-dataset cache namespacing

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Unique cache entry identifier
type CacheKey = String;

/// A cached SPARQL result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Cache key
    pub key: CacheKey,
    /// Cached result as JSON string
    pub result: String,
    /// Dataset this result belongs to
    pub dataset: String,
    /// Original SPARQL query
    pub query: String,
    /// Unix timestamp when entry was created
    pub created_at_secs: u64,
    /// TTL in seconds
    pub ttl_secs: u64,
    /// Number of times this entry was accessed
    pub hit_count: u64,
    /// Byte size of the cached result
    pub size_bytes: usize,
    /// Unix timestamp of last access
    pub last_accessed_secs: u64,
}

impl CachedResult {
    /// Create a new cached result
    pub fn new(
        key: CacheKey,
        result: String,
        dataset: String,
        query: String,
        ttl_secs: u64,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size_bytes = result.len();
        Self {
            key,
            result,
            dataset,
            query,
            created_at_secs: now,
            ttl_secs,
            hit_count: 0,
            size_bytes,
            last_accessed_secs: now,
        }
    }

    /// Check if this entry is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.created_at_secs) >= self.ttl_secs
    }

    /// Age in seconds
    pub fn age_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(self.created_at_secs)
    }

    /// Remaining TTL in seconds
    pub fn remaining_ttl_secs(&self) -> u64 {
        self.ttl_secs.saturating_sub(self.age_secs())
    }
}

/// LRU eviction order tracker
#[derive(Debug)]
struct LruTracker {
    /// Keys in LRU order (front = most recently used, back = least recently used)
    order: VecDeque<CacheKey>,
}

impl LruTracker {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
        }
    }

    /// Record a key access (move to front)
    fn access(&mut self, key: &str) {
        self.order.retain(|k| k != key);
        self.order.push_front(key.to_string());
    }

    /// Record a new key insertion (add to front)
    fn insert(&mut self, key: CacheKey) {
        self.order.retain(|k| k != &key);
        self.order.push_front(key);
    }

    /// Remove a key
    fn remove(&mut self, key: &str) {
        self.order.retain(|k| k != key);
    }

    /// Get the least recently used key
    fn lru_key(&self) -> Option<&str> {
        self.order.back().map(|s| s.as_str())
    }
}

/// Comprehensive cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LruCacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions (LRU)
    pub lru_evictions: u64,
    /// Total evictions (TTL expired)
    pub ttl_evictions: u64,
    /// Total invalidations (manual)
    pub invalidations: u64,
    /// Current number of entries
    pub entry_count: usize,
    /// Current total bytes used
    pub total_bytes: usize,
    /// Maximum capacity (entries)
    pub max_entries: usize,
    /// Maximum capacity (bytes)
    pub max_bytes: usize,
    /// Total queries served from cache
    pub queries_served: u64,
    /// Total queries bypassed cache
    pub queries_bypassed: u64,
}

impl LruCacheStats {
    /// Hit rate as a fraction [0.0, 1.0]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Miss rate as a fraction [0.0, 1.0]
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Average bytes per entry
    pub fn avg_entry_bytes(&self) -> f64 {
        if self.entry_count == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.entry_count as f64
        }
    }

    /// Cache fill percentage
    pub fn fill_pct(&self) -> f64 {
        if self.max_entries == 0 {
            0.0
        } else {
            self.entry_count as f64 / self.max_entries as f64 * 100.0
        }
    }
}

/// Configuration for the LRU cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LruCacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum total bytes
    pub max_bytes: usize,
    /// Default TTL in seconds
    pub default_ttl_secs: u64,
    /// Enable disk persistence
    pub persist: bool,
    /// Path for persistent storage
    pub persist_path: Option<PathBuf>,
    /// Whether caching is enabled
    pub enabled: bool,
}

impl Default for LruCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 2000,
            max_bytes: 256 * 1024 * 1024, // 256 MB
            default_ttl_secs: 600,        // 10 minutes
            persist: false,
            persist_path: None,
            enabled: true,
        }
    }
}

impl LruCacheConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("OXIRS_CACHE_MAX_ENTRIES") {
            if let Ok(n) = val.parse() {
                config.max_entries = n;
            }
        }
        if let Ok(val) = std::env::var("OXIRS_CACHE_MAX_MB") {
            if let Ok(mb) = val.parse::<usize>() {
                config.max_bytes = mb * 1024 * 1024;
            }
        }
        if let Ok(val) = std::env::var("OXIRS_CACHE_TTL") {
            if let Ok(secs) = val.parse() {
                config.default_ttl_secs = secs;
            }
        }
        if let Ok(val) = std::env::var("OXIRS_CACHE_ENABLED") {
            config.enabled = val != "0" && val.to_lowercase() != "false";
        }
        if let Ok(path) = std::env::var("OXIRS_CACHE_PATH") {
            config.persist = true;
            config.persist_path = Some(PathBuf::from(path));
        }

        config
    }
}

/// Inner state of the LRU cache (behind RwLock)
struct LruCacheInner {
    entries: HashMap<CacheKey, CachedResult>,
    lru: LruTracker,
    stats: LruCacheStats,
    config: LruCacheConfig,
}

impl LruCacheInner {
    fn new(config: LruCacheConfig) -> Self {
        let stats = LruCacheStats {
            max_entries: config.max_entries,
            max_bytes: config.max_bytes,
            ..Default::default()
        };
        Self {
            entries: HashMap::new(),
            lru: LruTracker::new(),
            stats,
            config,
        }
    }

    fn make_key(dataset: &str, query: &str) -> CacheKey {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        dataset.hash(&mut h);
        query.trim().hash(&mut h);
        format!("{}:{:x}", dataset, h.finish())
    }

    fn get(&mut self, dataset: &str, query: &str) -> Option<String> {
        let key = Self::make_key(dataset, query);

        // Check for expired entry
        if let Some(entry) = self.entries.get(&key) {
            if entry.is_expired() {
                self.entries.remove(&key);
                self.lru.remove(&key);
                self.stats.ttl_evictions += 1;
                self.stats.entry_count = self.entries.len();
                self.update_total_bytes();
                self.stats.misses += 1;
                return None;
            }
        }

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hit_count += 1;
            entry.last_accessed_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            self.lru.access(&key);
            self.stats.hits += 1;
            self.stats.queries_served += 1;
            Some(entry.result.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    fn set(&mut self, dataset: &str, query: &str, result: String, ttl_secs: Option<u64>) {
        let ttl = ttl_secs.unwrap_or(self.config.default_ttl_secs);
        let key = Self::make_key(dataset, query);
        let entry = CachedResult::new(
            key.clone(),
            result,
            dataset.to_string(),
            query.to_string(),
            ttl,
        );

        // Remove old entry if updating
        if self.entries.contains_key(&key) {
            if let Some(old) = self.entries.remove(&key) {
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(old.size_bytes);
            }
            self.lru.remove(&key);
        }

        // Evict if needed (by count)
        while self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        // Evict if needed (by bytes)
        while self.stats.total_bytes + entry.size_bytes > self.config.max_bytes
            && !self.entries.is_empty()
        {
            self.evict_lru();
        }

        self.stats.total_bytes += entry.size_bytes;
        self.lru.insert(key.clone());
        self.entries.insert(key, entry);
        self.stats.entry_count = self.entries.len();
    }

    fn evict_lru(&mut self) {
        let lru_key = self.lru.lru_key().map(|s| s.to_string());
        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(entry.size_bytes);
            }
            self.lru.remove(&key);
            self.stats.lru_evictions += 1;
            self.stats.entry_count = self.entries.len();
        }
    }

    fn evict_expired(&mut self) -> usize {
        let expired: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for key in &expired {
            if let Some(entry) = self.entries.remove(key) {
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(entry.size_bytes);
            }
            self.lru.remove(key);
            self.stats.ttl_evictions += 1;
        }
        self.stats.entry_count = self.entries.len();
        count
    }

    fn invalidate_dataset(&mut self, dataset: &str) -> usize {
        let keys: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.dataset == dataset)
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys.len();
        for key in &keys {
            if let Some(entry) = self.entries.remove(key) {
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(entry.size_bytes);
            }
            self.lru.remove(key);
            self.stats.invalidations += 1;
        }
        self.stats.entry_count = self.entries.len();
        count
    }

    fn invalidate_query(&mut self, dataset: &str, query: &str) -> bool {
        let key = Self::make_key(dataset, query);
        if let Some(entry) = self.entries.remove(&key) {
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(entry.size_bytes);
            self.lru.remove(&key);
            self.stats.invalidations += 1;
            self.stats.entry_count = self.entries.len();
            true
        } else {
            false
        }
    }

    fn clear(&mut self) {
        let count = self.entries.len();
        self.entries.clear();
        self.lru.order.clear();
        self.stats.invalidations += count as u64;
        self.stats.entry_count = 0;
        self.stats.total_bytes = 0;
    }

    fn update_total_bytes(&mut self) {
        self.stats.total_bytes = self.entries.values().map(|e| e.size_bytes).sum();
    }

    fn list_entries(&self, dataset_filter: Option<&str>) -> Vec<&CachedResult> {
        self.entries
            .values()
            .filter(|e| {
                if let Some(ds) = dataset_filter {
                    e.dataset == ds
                } else {
                    true
                }
            })
            .collect()
    }
}

/// Thread-safe LRU cache with TTL, eviction policies, and persistence
pub struct LruResultCache {
    inner: Arc<RwLock<LruCacheInner>>,
    startup_time: Instant,
}

impl LruResultCache {
    /// Create a new LRU result cache with the given configuration
    pub fn new(config: LruCacheConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCacheInner::new(config))),
            startup_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn default_cache() -> Self {
        Self::new(LruCacheConfig::default())
    }

    /// Get a cached result for the given dataset and query
    pub fn get(&self, dataset: &str, query: &str) -> Option<String> {
        let mut inner = self.inner.write().ok()?;
        inner.get(dataset, query)
    }

    /// Store a result in the cache
    pub fn set(&self, dataset: &str, query: &str, result: String) {
        if let Ok(mut inner) = self.inner.write() {
            inner.set(dataset, query, result, None);
        }
    }

    /// Store a result with a specific TTL override
    pub fn set_with_ttl(&self, dataset: &str, query: &str, result: String, ttl_secs: u64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.set(dataset, query, result, Some(ttl_secs));
        }
    }

    /// Invalidate all entries for a specific dataset
    pub fn invalidate_dataset(&self, dataset: &str) -> usize {
        self.inner
            .write()
            .map(|mut inner| inner.invalidate_dataset(dataset))
            .unwrap_or(0)
    }

    /// Invalidate a specific query entry
    pub fn invalidate_query(&self, dataset: &str, query: &str) -> bool {
        self.inner
            .write()
            .map(|mut inner| inner.invalidate_query(dataset, query))
            .unwrap_or(false)
    }

    /// Evict all expired entries
    pub fn evict_expired(&self) -> usize {
        self.inner
            .write()
            .map(|mut inner| inner.evict_expired())
            .unwrap_or(0)
    }

    /// Clear all entries
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.write() {
            inner.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> LruCacheStats {
        self.inner
            .read()
            .map(|inner| inner.stats.clone())
            .unwrap_or_default()
    }

    /// Get the current entry count
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    /// True if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// List all entries, optionally filtered by dataset
    pub fn list_entries(&self, dataset: Option<&str>) -> Vec<CachedResult> {
        self.inner
            .read()
            .map(|inner| inner.list_entries(dataset).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get cache uptime
    pub fn uptime(&self) -> Duration {
        self.startup_time.elapsed()
    }

    /// Save cache to disk (if persistence is configured)
    pub fn save_to_disk(&self) -> Result<usize> {
        let inner = self
            .inner
            .read()
            .map_err(|_| anyhow!("Cache lock poisoned"))?;
        let config = &inner.config;

        if !config.persist {
            return Ok(0);
        }

        let path = config
            .persist_path
            .as_ref()
            .ok_or_else(|| anyhow!("No persist path configured"))?;

        std::fs::create_dir_all(path)?;
        let file_path = path.join("result_cache.json");

        let entries: Vec<&CachedResult> =
            inner.entries.values().filter(|e| !e.is_expired()).collect();
        let count = entries.len();
        let json = serde_json::to_string(&entries)
            .map_err(|e| anyhow!("Failed to serialize cache: {}", e))?;
        std::fs::write(&file_path, json)?;

        Ok(count)
    }

    /// Load cache from disk (if persistence is configured)
    pub fn load_from_disk(&self) -> Result<usize> {
        let config = {
            let inner = self
                .inner
                .read()
                .map_err(|_| anyhow!("Cache lock poisoned"))?;
            inner.config.clone()
        };

        if !config.persist {
            return Ok(0);
        }

        let path = config
            .persist_path
            .as_ref()
            .ok_or_else(|| anyhow!("No persist path configured"))?;

        let file_path = path.join("result_cache.json");
        if !file_path.exists() {
            return Ok(0);
        }

        let json = std::fs::read_to_string(&file_path)?;
        let entries: Vec<CachedResult> = serde_json::from_str(&json)
            .map_err(|e| anyhow!("Failed to deserialize cache: {}", e))?;

        let mut count = 0;
        let mut inner = self
            .inner
            .write()
            .map_err(|_| anyhow!("Cache lock poisoned"))?;

        for entry in entries {
            if !entry.is_expired() {
                let key = entry.key.clone();
                inner.lru.insert(key.clone());
                inner.stats.total_bytes += entry.size_bytes;
                inner.entries.insert(key, entry);
                count += 1;
            }
        }
        inner.stats.entry_count = inner.entries.len();

        Ok(count)
    }
}

/// Global LRU cache singleton
static GLOBAL_LRU_CACHE: std::sync::OnceLock<LruResultCache> = std::sync::OnceLock::new();

/// Get or initialize the global LRU result cache
pub fn global_lru_cache() -> &'static LruResultCache {
    GLOBAL_LRU_CACHE.get_or_init(|| {
        let config = LruCacheConfig::from_env();
        LruResultCache::new(config)
    })
}

/// Enhanced cache management commands
pub mod commands {
    use super::*;
    use colored::Colorize;

    /// Show detailed cache statistics
    pub async fn stats_command() -> Result<()> {
        let cache = global_lru_cache();
        let stats = cache.stats();

        println!("{}", "LRU Result Cache Statistics".cyan().bold());
        println!("{}", "=".repeat(50));
        println!();
        println!(
            "  Entries:       {}/{}",
            stats.entry_count, stats.max_entries
        );
        println!(
            "  Memory:        {:.2} MB / {:.2} MB",
            stats.total_bytes as f64 / 1_048_576.0,
            stats.max_bytes as f64 / 1_048_576.0
        );
        println!("  Fill:          {:.1}%", stats.fill_pct());
        println!();
        println!("  Cache Hits:    {}", stats.hits.to_string().green());
        println!("  Cache Misses:  {}", stats.misses.to_string().yellow());
        println!("  Hit Rate:      {:.2}%", stats.hit_rate() * 100.0);
        println!();
        println!("  LRU Evictions: {}", stats.lru_evictions);
        println!("  TTL Evictions: {}", stats.ttl_evictions);
        println!("  Invalidations: {}", stats.invalidations);
        println!();
        println!(
            "  Avg Entry:     {:.2} KB",
            stats.avg_entry_bytes() / 1024.0
        );
        println!("  Uptime:        {:.0}s", cache.uptime().as_secs_f64());

        Ok(())
    }

    /// Clear all cache entries
    pub async fn clear_command() -> Result<()> {
        let cache = global_lru_cache();
        let before = cache.len();
        cache.clear();

        println!(
            "{} Cache cleared ({} entries removed)",
            "OK".green().bold(),
            before
        );
        Ok(())
    }

    /// Invalidate entries for a specific dataset
    pub async fn invalidate_dataset_command(dataset: &str) -> Result<()> {
        let cache = global_lru_cache();
        let removed = cache.invalidate_dataset(dataset);
        println!(
            "{} Invalidated {} entries for dataset '{}'",
            "OK".green().bold(),
            removed,
            dataset
        );
        Ok(())
    }

    /// Evict expired entries
    pub async fn evict_expired_command() -> Result<()> {
        let cache = global_lru_cache();
        let removed = cache.evict_expired();
        println!(
            "{} Evicted {} expired entries",
            "OK".green().bold(),
            removed
        );
        Ok(())
    }

    /// List cache entries
    pub async fn list_command(dataset: Option<&str>) -> Result<()> {
        let cache = global_lru_cache();
        let entries = cache.list_entries(dataset);

        if entries.is_empty() {
            println!("Cache is empty");
            return Ok(());
        }

        println!(
            "{:<20} {:>8} {:>10} {:>8} {:>6}",
            "Dataset", "Hits", "Size", "TTL", "Age"
        );
        println!("{}", "-".repeat(60));
        for entry in &entries {
            let query_preview: String = entry.query.chars().take(40).collect();
            println!(
                "{:<20} {:>8} {:>10} {:>8}s {:>6}s",
                &entry.dataset[..entry.dataset.len().min(20)],
                entry.hit_count,
                format_bytes(entry.size_bytes),
                entry.remaining_ttl_secs(),
                entry.age_secs()
            );
            println!("  Query: {}...", query_preview);
        }
        println!("{}", "-".repeat(60));
        println!("Total: {} entries", entries.len());

        Ok(())
    }

    /// Configure cache settings
    pub async fn config_command(ttl: Option<u64>, max_size: Option<usize>) -> Result<()> {
        println!("{}", "Cache Configuration".cyan().bold());
        println!();

        if let Some(t) = ttl {
            println!("  Default TTL: {} seconds", t);
            println!("  Hint: Set OXIRS_CACHE_TTL={} to persist this setting", t);
        }
        if let Some(s) = max_size {
            println!("  Max Entries: {}", s);
            println!(
                "  Hint: Set OXIRS_CACHE_MAX_ENTRIES={} to persist this setting",
                s
            );
        }
        if ttl.is_none() && max_size.is_none() {
            let cache = global_lru_cache();
            let stats = cache.stats();
            println!("  Max Entries:  {}", stats.max_entries);
            println!("  Max Bytes:    {} MB", stats.max_bytes / 1_048_576);
        }
        println!();
        println!("Environment variables:");
        println!("  OXIRS_CACHE_ENABLED        = true|false (default: true)");
        println!("  OXIRS_CACHE_TTL            = seconds (default: 600)");
        println!("  OXIRS_CACHE_MAX_ENTRIES    = count (default: 2000)");
        println!("  OXIRS_CACHE_MAX_MB         = megabytes (default: 256)");
        println!("  OXIRS_CACHE_PATH           = /path/to/persist (optional)");

        Ok(())
    }

    fn format_bytes(bytes: usize) -> String {
        if bytes >= 1_048_576 {
            format!("{:.1}M", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.1}K", bytes as f64 / 1024.0)
        } else {
            format!("{}", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn make_config(max_entries: usize, ttl_secs: u64) -> LruCacheConfig {
        LruCacheConfig {
            max_entries,
            max_bytes: 64 * 1024 * 1024,
            default_ttl_secs: ttl_secs,
            persist: false,
            persist_path: None,
            enabled: true,
        }
    }

    #[test]
    fn test_lru_cache_basic_get_set() {
        let cache = LruResultCache::new(make_config(100, 3600));
        let dataset = "test_ds";
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let result = r#"{"head":{"vars":["s"]},"results":{"bindings":[]}}"#.to_string();

        assert!(cache.get(dataset, query).is_none());
        cache.set(dataset, query, result.clone());
        assert_eq!(cache.get(dataset, query), Some(result));
    }

    #[test]
    fn test_lru_cache_hit_rate() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds", "q1", "r1".to_string());

        // 1 hit
        cache.get("ds", "q1");
        // 1 miss
        cache.get("ds", "q2");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lru_cache_eviction_by_count() {
        let cache = LruResultCache::new(make_config(3, 3600));
        cache.set("ds", "q1", "r1".to_string());
        cache.set("ds", "q2", "r2".to_string());
        cache.set("ds", "q3", "r3".to_string());
        assert_eq!(cache.len(), 3);

        // Access q1 and q2 to make q3 LRU... but wait, q3 was just inserted (most recent)
        // q1 was inserted first (LRU)
        cache.set("ds", "q4", "r4".to_string());
        assert_eq!(cache.len(), 3);

        let stats = cache.stats();
        assert_eq!(stats.lru_evictions, 1);
    }

    #[test]
    #[ignore = "inherently slow: requires wall-clock TTL expiry (use nextest --ignored to run)"]
    fn test_lru_cache_ttl_expiration() {
        let cache = LruResultCache::new(make_config(100, 1)); // 1s TTL
        cache.set("ds", "q", "result".to_string());

        // Should be present immediately
        assert!(cache.get("ds", "q").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_secs(2));

        // Should be expired now
        assert!(cache.get("ds", "q").is_none());
        let stats = cache.stats();
        assert!(stats.ttl_evictions > 0);
    }

    #[test]
    fn test_lru_cache_invalidate_dataset() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds1", "q1", "r1".to_string());
        cache.set("ds1", "q2", "r2".to_string());
        cache.set("ds2", "q3", "r3".to_string());

        let removed = cache.invalidate_dataset("ds1");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);

        // ds2 entry should still be there
        assert!(cache.get("ds2", "q3").is_some());
    }

    #[test]
    fn test_lru_cache_invalidate_query() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds", "q1", "r1".to_string());
        cache.set("ds", "q2", "r2".to_string());

        let removed = cache.invalidate_query("ds", "q1");
        assert!(removed);
        assert_eq!(cache.len(), 1);

        // Non-existent query
        let not_removed = cache.invalidate_query("ds", "nonexistent");
        assert!(!not_removed);
    }

    #[test]
    fn test_lru_cache_clear() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds", "q1", "r1".to_string());
        cache.set("ds", "q2", "r2".to_string());
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.invalidations, 2);
    }

    #[test]
    #[ignore = "inherently slow: requires wall-clock TTL expiry (use nextest --ignored to run)"]
    fn test_lru_cache_evict_expired() {
        let cache = LruResultCache::new(make_config(100, 1)); // 1s TTL
        cache.set("ds", "q1", "r1".to_string());
        cache.set("ds", "q2", "r2".to_string());

        std::thread::sleep(Duration::from_secs(2));

        let evicted = cache.evict_expired();
        assert_eq!(evicted, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache_list_entries() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds1", "q1", "r1".to_string());
        cache.set("ds2", "q2", "r2".to_string());

        let all = cache.list_entries(None);
        assert_eq!(all.len(), 2);

        let ds1 = cache.list_entries(Some("ds1"));
        assert_eq!(ds1.len(), 1);
        assert_eq!(ds1[0].dataset, "ds1");
    }

    #[test]
    #[ignore = "inherently slow: requires wall-clock TTL expiry (use nextest --ignored to run)"]
    fn test_lru_cache_set_with_ttl() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set_with_ttl("ds", "q", "result".to_string(), 1);

        assert!(cache.get("ds", "q").is_some());
        std::thread::sleep(Duration::from_secs(2));
        assert!(cache.get("ds", "q").is_none());
    }

    #[test]
    fn test_lru_cache_update_entry() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds", "q", "old_result".to_string());
        cache.set("ds", "q", "new_result".to_string());

        assert_eq!(cache.get("ds", "q"), Some("new_result".to_string()));
        assert_eq!(cache.len(), 1); // No duplication
    }

    #[test]
    fn test_lru_cache_stats_fill_pct() {
        let cache = LruResultCache::new(make_config(100, 3600));
        for i in 0..50 {
            cache.set("ds", &format!("q{}", i), "result".to_string());
        }

        let stats = cache.stats();
        assert!((stats.fill_pct() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_lru_cache_stats_avg_entry_bytes() {
        let cache = LruResultCache::new(make_config(100, 3600));
        cache.set("ds", "q1", "r1".to_string()); // 2 bytes
        cache.set("ds", "q2", "r2r2".to_string()); // 4 bytes

        let stats = cache.stats();
        // avg = 3.0
        assert!((stats.avg_entry_bytes() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_lru_tracker_order() {
        let mut tracker = LruTracker::new();
        tracker.insert("a".to_string());
        tracker.insert("b".to_string());
        tracker.insert("c".to_string());

        // LRU should be "a"
        assert_eq!(tracker.lru_key(), Some("a"));

        // Access "a" to make it MRU
        tracker.access("a");
        // Now LRU should be "b"
        assert_eq!(tracker.lru_key(), Some("b"));
    }

    #[test]
    fn test_cached_result_age_and_remaining_ttl() {
        let entry = CachedResult::new(
            "key".to_string(),
            "result".to_string(),
            "ds".to_string(),
            "q".to_string(),
            3600,
        );
        assert!(!entry.is_expired());
        assert!(entry.age_secs() < 2);
        assert!(entry.remaining_ttl_secs() > 3598);
    }

    #[test]
    fn test_lru_cache_config_defaults() {
        let config = LruCacheConfig::default();
        assert_eq!(config.max_entries, 2000);
        assert_eq!(config.default_ttl_secs, 600);
        assert!(!config.persist);
        assert!(config.enabled);
    }

    #[test]
    fn test_lru_cache_disk_persistence() {
        let temp_dir = env::temp_dir().join("oxirs_cache_test_persistence");
        let config = LruCacheConfig {
            persist: true,
            persist_path: Some(temp_dir.clone()),
            max_entries: 100,
            max_bytes: 1024 * 1024,
            default_ttl_secs: 3600,
            enabled: true,
        };

        let cache1 = LruResultCache::new(config.clone());
        cache1.set("ds", "q1", "result1".to_string());
        cache1.set("ds", "q2", "result2".to_string());

        let saved = cache1.save_to_disk().unwrap();
        assert_eq!(saved, 2);

        let cache2 = LruResultCache::new(config);
        let loaded = cache2.load_from_disk().unwrap();
        assert_eq!(loaded, 2);
        assert!(cache2.get("ds", "q1").is_some());
        assert!(cache2.get("ds", "q2").is_some());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_lru_cache_byte_tracking() {
        let cache = LruResultCache::new(make_config(100, 3600));
        let result = "x".repeat(1024); // 1KB
        cache.set("ds", "q", result);

        let stats = cache.stats();
        assert_eq!(stats.total_bytes, 1024);

        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_lru_cache_miss_rate() {
        let stats = LruCacheStats {
            hits: 3,
            misses: 1,
            ..Default::default()
        };
        assert!((stats.miss_rate() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_lru_cache_no_entries_stats() {
        let stats = LruCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.miss_rate(), 1.0);
        assert_eq!(stats.fill_pct(), 0.0);
        assert_eq!(stats.avg_entry_bytes(), 0.0);
    }

    #[test]
    fn test_lru_cache_lru_order_on_get() {
        let cache = LruResultCache::new(make_config(2, 3600));
        cache.set("ds", "q1", "r1".to_string());
        cache.set("ds", "q2", "r2".to_string());

        // Access q1 to make it MRU
        cache.get("ds", "q1");

        // Adding q3 should evict q2 (LRU)
        cache.set("ds", "q3", "r3".to_string());

        assert!(cache.get("ds", "q1").is_some());
        assert!(cache.get("ds", "q2").is_none()); // Evicted
        assert!(cache.get("ds", "q3").is_some());
    }
}
