//! Query Result Caching with Intelligent Invalidation
//!
//! This module provides high-performance query result caching with:
//! - LRU eviction policy with TTL support
//! - Intelligent cache invalidation on updates
//! - Query normalization for better cache hit rates
//! - Memory-aware cache sizing
//! - Cache statistics and monitoring
//! - Graph-based invalidation tracking

use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// Re-export std AtomicU64 to avoid conflicts with metrics::atomics::AtomicU64
use std::sync::atomic::AtomicU64 as StdAtomicU64;

/// Cached query result
#[derive(Debug)]
pub struct CachedResult {
    /// Query result (serialized)
    pub result: Arc<Vec<u8>>,
    /// Result size in bytes
    pub size_bytes: usize,
    /// Timestamp when cached
    pub cached_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,
    /// Access count
    pub access_count: StdAtomicU64,
    /// Last accessed timestamp
    pub last_accessed: Arc<RwLock<DateTime<Utc>>>,
    /// Graphs referenced by this query
    pub referenced_graphs: HashSet<String>,
    /// Query type (SELECT, CONSTRUCT, etc.)
    pub query_type: QueryType,
}

/// Query type for cache strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Default TTL for cached results
    pub default_ttl: Duration,
    /// Enable automatic cleanup
    pub enable_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Enable query normalization
    pub enable_normalization: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            max_entries: 10_000,
            default_ttl: Duration::minutes(10),
            enable_cleanup: true,
            cleanup_interval_secs: 60,
            enable_normalization: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
    pub entry_count: usize,
    pub size_bytes: usize,
    pub size_mb: f64,
    pub eviction_count: u64,
    pub invalidation_count: u64,
    pub avg_entry_size: f64,
    pub memory_usage_percent: f64,
}

/// Query result cache with intelligent invalidation
pub struct QueryCache {
    /// Cache entries (query hash -> cached result)
    cache: Arc<DashMap<String, CachedResult>>,
    /// Graph to query mapping (graph -> set of query hashes)
    graph_queries: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Configuration
    config: CacheConfig,
    /// Current cache size in bytes
    current_size: Arc<AtomicUsize>,
    /// Cache hits counter
    hits: Arc<AtomicU64>,
    /// Cache misses counter
    misses: Arc<AtomicU64>,
    /// Evictions counter
    evictions: Arc<AtomicU64>,
    /// Invalidations counter
    invalidations: Arc<AtomicU64>,
}

impl QueryCache {
    /// Create new query cache
    pub fn new(config: CacheConfig) -> Self {
        let cache = QueryCache {
            cache: Arc::new(DashMap::new()),
            graph_queries: Arc::new(RwLock::new(HashMap::new())),
            config,
            current_size: Arc::new(AtomicUsize::new(0)),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
            invalidations: Arc::new(AtomicU64::new(0)),
        };

        cache
    }

    /// Get cached result for a query
    pub async fn get(&self, query: &str, graphs: &[String]) -> Option<Vec<u8>> {
        let query_hash = self.compute_query_hash(query);

        if let Some(entry) = self.cache.get(&query_hash) {
            // Check expiration
            if Utc::now() > entry.expires_at {
                drop(entry); // Release read lock
                self.remove(&query_hash).await;
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Check if graphs match
            if !graphs.is_empty() && !self.graphs_match(&entry.referenced_graphs, graphs) {
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access stats
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            *entry.last_accessed.write().await = Utc::now();

            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!("Cache hit for query hash: {}", query_hash);

            Some((*entry.result).clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            debug!("Cache miss for query hash: {}", query_hash);
            None
        }
    }

    /// Cache a query result
    pub async fn put(
        &self,
        query: &str,
        result: Vec<u8>,
        query_type: QueryType,
        referenced_graphs: HashSet<String>,
        ttl: Option<Duration>,
    ) {
        let query_hash = self.compute_query_hash(query);
        let size_bytes = result.len();

        // Check if we need to evict entries
        self.ensure_capacity(size_bytes).await;

        let ttl = ttl.unwrap_or(self.config.default_ttl);

        let cached_result = CachedResult {
            result: Arc::new(result),
            size_bytes,
            cached_at: Utc::now(),
            expires_at: Utc::now() + ttl,
            access_count: StdAtomicU64::new(0),
            last_accessed: Arc::new(RwLock::new(Utc::now())),
            referenced_graphs: referenced_graphs.clone(),
            query_type,
        };

        // Update graph mappings
        {
            let mut graph_queries = self.graph_queries.write().await;
            for graph in &referenced_graphs {
                graph_queries
                    .entry(graph.clone())
                    .or_insert_with(HashSet::new)
                    .insert(query_hash.clone());
            }
        }

        self.cache.insert(query_hash.clone(), cached_result);
        self.current_size.fetch_add(size_bytes, Ordering::Relaxed);

        debug!(
            "Cached query result: {} bytes (TTL: {}s)",
            size_bytes,
            ttl.num_seconds()
        );
    }

    /// Invalidate cache entries for specific graphs
    pub async fn invalidate_graphs(&self, graphs: &[String]) {
        let mut invalidated = 0;

        let queries_to_invalidate: HashSet<String> = {
            let graph_queries = self.graph_queries.read().await;
            graphs
                .iter()
                .flat_map(|g| graph_queries.get(g).cloned().unwrap_or_default())
                .collect()
        };

        for query_hash in queries_to_invalidate {
            self.remove(&query_hash).await;
            invalidated += 1;
        }

        // Clean up graph mappings
        {
            let mut graph_queries = self.graph_queries.write().await;
            for graph in graphs {
                graph_queries.remove(graph);
            }
        }

        self.invalidations.fetch_add(invalidated, Ordering::Relaxed);

        if invalidated > 0 {
            info!(
                "Invalidated {} cache entries for {} graphs",
                invalidated,
                graphs.len()
            );
        }
    }

    /// Invalidate all cache entries
    pub async fn invalidate_all(&self) {
        let count = self.cache.len();
        self.cache.clear();
        self.graph_queries.write().await.clear();
        self.current_size.store(0, Ordering::Relaxed);
        self.invalidations
            .fetch_add(count as u64, Ordering::Relaxed);

        info!("Invalidated all {} cache entries", count);
    }

    /// Remove a specific cache entry
    async fn remove(&self, query_hash: &str) {
        if let Some((_, entry)) = self.cache.remove(query_hash) {
            self.current_size
                .fetch_sub(entry.size_bytes, Ordering::Relaxed);

            // Remove from graph mappings
            let mut graph_queries = self.graph_queries.write().await;
            for graph in &entry.referenced_graphs {
                if let Some(queries) = graph_queries.get_mut(graph) {
                    queries.remove(query_hash);
                    if queries.is_empty() {
                        graph_queries.remove(graph);
                    }
                }
            }
        }
    }

    /// Ensure cache has capacity for new entry
    async fn ensure_capacity(&self, needed_bytes: usize) {
        let current_size = self.current_size.load(Ordering::Relaxed);
        let current_count = self.cache.len();

        // Check if we need to evict
        if current_size + needed_bytes > self.config.max_size_bytes
            || current_count >= self.config.max_entries
        {
            self.evict_lru(needed_bytes).await;
        }
    }

    /// Evict least recently used entries
    async fn evict_lru(&self, needed_bytes: usize) {
        let mut entries_to_evict: Vec<(String, DateTime<Utc>, usize)> = Vec::new();

        // Collect LRU entries
        for entry in self.cache.iter() {
            let last_accessed = *entry.value().last_accessed.read().await;
            entries_to_evict.push((entry.key().clone(), last_accessed, entry.value().size_bytes));
        }

        // Sort by last accessed (oldest first)
        entries_to_evict.sort_by_key(|(_, accessed, _)| *accessed);

        let mut freed_bytes = 0;
        let mut evicted_count = 0;

        for (query_hash, _, size) in entries_to_evict {
            if freed_bytes >= needed_bytes && self.cache.len() < self.config.max_entries * 9 / 10 {
                break; // Freed enough space and below 90% capacity
            }

            self.remove(&query_hash).await;
            freed_bytes += size;
            evicted_count += 1;
        }

        self.evictions.fetch_add(evicted_count, Ordering::Relaxed);

        if evicted_count > 0 {
            debug!(
                "Evicted {} entries, freed {} bytes",
                evicted_count, freed_bytes
            );
        }
    }

    /// Cleanup expired entries
    pub async fn cleanup_expired(&self) {
        let now = Utc::now();
        let mut expired = Vec::new();

        for entry in self.cache.iter() {
            if now > entry.value().expires_at {
                expired.push(entry.key().clone());
            }
        }

        let count = expired.len();

        for query_hash in &expired {
            self.remove(query_hash).await;
        }

        if count > 0 {
            debug!("Cleaned up {} expired cache entries", count);
        }
    }

    /// Compute normalized query hash
    fn compute_query_hash(&self, query: &str) -> String {
        use sha2::{Digest, Sha256};

        let normalized = if self.config.enable_normalization {
            self.normalize_query(query)
        } else {
            query.to_string()
        };

        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Normalize SPARQL query for better cache hits
    fn normalize_query(&self, query: &str) -> String {
        // Remove comments
        let mut normalized = query
            .lines()
            .filter(|line| !line.trim().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");

        // Normalize whitespace
        normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

        // Convert to lowercase (except for literals)
        // This is a simplified version - production would use proper SPARQL parsing
        normalized.to_lowercase()
    }

    /// Check if graphs match
    fn graphs_match(&self, cached_graphs: &HashSet<String>, query_graphs: &[String]) -> bool {
        if query_graphs.is_empty() {
            return true; // No graph restriction
        }

        query_graphs
            .iter()
            .all(|g| cached_graphs.contains(g) || cached_graphs.is_empty())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        let size_bytes = self.current_size.load(Ordering::Relaxed);
        let entry_count = self.cache.len();
        let avg_entry_size = if entry_count > 0 {
            size_bytes as f64 / entry_count as f64
        } else {
            0.0
        };

        let memory_usage_percent = (size_bytes as f64 / self.config.max_size_bytes as f64) * 100.0;

        CacheStats {
            total_hits: hits,
            total_misses: misses,
            hit_rate,
            entry_count,
            size_bytes,
            size_mb: size_bytes as f64 / (1024.0 * 1024.0),
            eviction_count: self.evictions.load(Ordering::Relaxed),
            invalidation_count: self.invalidations.load(Ordering::Relaxed),
            avg_entry_size,
            memory_usage_percent,
        }
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(self: Arc<Self>) {
        if !self.config.enable_cleanup {
            return;
        }

        let cache = Arc::clone(&self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                cache.config.cleanup_interval_secs,
            ));

            loop {
                interval.tick().await;
                cache.cleanup_expired().await;
            }
        });

        info!(
            "Started cache cleanup task (interval: {}s)",
            self.config.cleanup_interval_secs
        );
    }

    /// Warm up cache with common queries
    pub async fn warmup(&self, queries: Vec<(String, Vec<u8>, QueryType, HashSet<String>)>) {
        info!("Warming up cache with {} queries", queries.len());

        for (query, result, query_type, graphs) in queries {
            self.put(&query, result, query_type, graphs, None).await;
        }

        info!("Cache warmup complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> QueryCache {
        QueryCache::new(CacheConfig {
            max_size_bytes: 1024 * 1024, // 1MB for testing
            max_entries: 100,
            default_ttl: Duration::minutes(5),
            enable_cleanup: false,
            cleanup_interval_secs: 60,
            enable_normalization: true,
        })
    }

    #[tokio::test]
    async fn test_cache_put_and_get() {
        let cache = create_test_cache();

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let result = b"results".to_vec();

        cache
            .put(
                query,
                result.clone(),
                QueryType::Select,
                HashSet::new(),
                None,
            )
            .await;

        let cached = cache.get(query, &[]).await;
        assert_eq!(cached, Some(result));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = create_test_cache();

        let result = cache.get("SELECT * WHERE { ?s ?p ?o }", &[]).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let cache = create_test_cache();

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let result = b"results".to_vec();
        let mut graphs = HashSet::new();
        graphs.insert("http://example.org/graph1".to_string());

        cache
            .put(query, result, QueryType::Select, graphs, None)
            .await;

        // Verify cached
        assert!(cache.get(query, &[]).await.is_some());

        // Invalidate
        cache
            .invalidate_graphs(&["http://example.org/graph1".to_string()])
            .await;

        // Should be gone
        assert!(cache.get(query, &[]).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = create_test_cache();

        cache.get("query1", &[]).await; // Miss
        cache
            .put(
                "query2",
                b"result".to_vec(),
                QueryType::Select,
                HashSet::new(),
                None,
            )
            .await;
        cache.get("query2", &[]).await; // Hit

        let stats = cache.get_stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_query_normalization() {
        let cache = create_test_cache();

        let query1 = "SELECT * WHERE { ?s ?p ?o }";
        let query2 = "SELECT   *   WHERE   {   ?s   ?p   ?o   }"; // Extra whitespace

        let hash1 = cache.compute_query_hash(query1);
        let hash2 = cache.compute_query_hash(query2);

        assert_eq!(hash1, hash2);
    }

    #[tokio::test]
    async fn test_eviction() {
        let small_cache = QueryCache::new(CacheConfig {
            max_size_bytes: 100, // Very small
            max_entries: 5,
            default_ttl: Duration::minutes(5),
            enable_cleanup: false,
            cleanup_interval_secs: 60,
            enable_normalization: false,
        });

        // Fill cache beyond capacity
        for i in 0..10 {
            small_cache
                .put(
                    &format!("query{}", i),
                    vec![0u8; 50],
                    QueryType::Select,
                    HashSet::new(),
                    None,
                )
                .await;
        }

        let stats = small_cache.get_stats();
        assert!(stats.entry_count <= 5);
        assert!(stats.size_bytes <= 100);
    }
}
