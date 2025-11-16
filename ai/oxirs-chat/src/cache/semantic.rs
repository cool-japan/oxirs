//! Semantic Caching System
//!
//! Caches query results based on semantic similarity rather than exact matching.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

use scirs2_core::ndarray_ext::Array1;
// Note: cosine_similarity implementation available in crate::utils::stats if needed

/// Semantic cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Original query
    pub query: String,
    /// Query embedding
    #[serde(skip)]
    pub embedding: Option<Array1<f32>>,
    /// Cached result
    pub result: CachedResult,
    /// Cache timestamp
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub timestamp: Instant,
    /// Hit count
    pub hit_count: usize,
    /// Last access time
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub last_access: Instant,
    /// TTL (time-to-live)
    #[serde(skip)]
    #[serde(default = "default_ttl")]
    pub ttl: Duration,
}

fn default_ttl() -> Duration {
    Duration::from_secs(3600)
}

/// Cached result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    /// Result data (JSON-serialized)
    pub data: String,
    /// Result metadata
    pub metadata: HashMap<String, String>,
    /// Computation time in milliseconds (for cache efficiency metrics)
    pub computation_time_ms: u64,
}

/// Semantic cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Similarity threshold for cache hits (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Maximum cache size (number of entries)
    pub max_size: usize,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Enable embedding-based similarity
    pub use_embeddings: bool,
    /// Eviction strategy
    pub eviction_strategy: EvictionStrategy,
    /// Enable statistics tracking
    pub track_statistics: bool,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            max_size: 1000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            use_embeddings: true,
            eviction_strategy: EvictionStrategy::LRU,
            track_statistics: true,
        }
    }
}

/// Cache eviction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time To Live
    TTL,
    /// First In First Out
    FIFO,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Total queries processed
    pub total_queries: usize,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Average similarity score for hits
    pub avg_similarity: f32,
    /// Total time saved by caching
    pub time_saved: Duration,
    /// Current cache size
    pub current_size: usize,
    /// Eviction count
    pub evictions: usize,
}

/// Semantic cache
pub struct SemanticCache {
    config: SemanticCacheConfig,
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    statistics: Arc<RwLock<CacheStatistics>>,
}

impl SemanticCache {
    /// Create a new semantic cache
    pub fn new(config: SemanticCacheConfig) -> Self {
        info!(
            "Initialized semantic cache with similarity threshold: {}",
            config.similarity_threshold
        );

        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(CacheStatistics::default())),
        }
    }

    /// Get cached result for a query
    pub async fn get(&self, query: &str) -> Result<Option<CachedResult>> {
        debug!("Checking semantic cache for query: {}", query);

        let entries = self.entries.read().await;

        // Check for exact match first (fast path)
        if let Some(entry) = entries.get(query) {
            if !self.is_expired(entry) {
                self.record_hit(entry).await;
                return Ok(Some(entry.result.clone()));
            }
        }

        // Semantic similarity search (if embeddings enabled)
        if self.config.use_embeddings {
            let query_embedding = self.compute_embedding(query)?;

            for entry in entries.values() {
                if self.is_expired(entry) {
                    continue;
                }

                if let Some(ref entry_embedding) = entry.embedding {
                    let similarity = self.compute_similarity(&query_embedding, entry_embedding)?;

                    if similarity >= self.config.similarity_threshold {
                        debug!("Semantic cache hit with similarity: {:.3}", similarity);
                        self.record_hit_with_similarity(entry, similarity).await;
                        return Ok(Some(entry.result.clone()));
                    }
                }
            }
        }

        // Cache miss
        self.record_miss().await;
        Ok(None)
    }

    /// Put result in cache
    pub async fn put(
        &self,
        query: String,
        result: CachedResult,
        embedding: Option<Array1<f32>>,
    ) -> Result<()> {
        let mut entries = self.entries.write().await;

        // Evict if at capacity
        if entries.len() >= self.config.max_size {
            self.evict(&mut entries).await?;
        }

        let entry = CacheEntry {
            query: query.clone(),
            embedding,
            result,
            timestamp: Instant::now(),
            hit_count: 0,
            last_access: Instant::now(),
            ttl: self.config.default_ttl,
        };

        entries.insert(query, entry);
        self.update_size(entries.len()).await;

        Ok(())
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, query: &str) -> Result<bool> {
        let mut entries = self.entries.write().await;
        let removed = entries.remove(query).is_some();
        self.update_size(entries.len()).await;
        Ok(removed)
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.clear();
        self.update_size(0).await;
        Ok(())
    }

    /// Clean expired entries
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let mut entries = self.entries.write().await;
        let initial_size = entries.len();

        entries.retain(|_, entry| !self.is_expired(entry));

        let removed = initial_size - entries.len();
        self.update_size(entries.len()).await;

        if removed > 0 {
            info!("Cleaned up {} expired cache entries", removed);
        }

        Ok(removed)
    }

    /// Get cache statistics
    pub async fn statistics(&self) -> CacheStatistics {
        self.statistics.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_statistics(&self) -> Result<()> {
        let mut stats = self.statistics.write().await;
        *stats = CacheStatistics::default();
        Ok(())
    }

    // Helper methods

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        entry.timestamp.elapsed() > entry.ttl
    }

    async fn record_hit(&self, entry: &CacheEntry) {
        if !self.config.track_statistics {
            return;
        }

        let mut stats = self.statistics.write().await;
        stats.hits += 1;
        stats.total_queries += 1;
        stats.hit_rate = stats.hits as f64 / stats.total_queries as f64;
        stats.time_saved += Duration::from_millis(entry.result.computation_time_ms);
    }

    async fn record_hit_with_similarity(&self, entry: &CacheEntry, similarity: f32) {
        if !self.config.track_statistics {
            return;
        }

        let mut stats = self.statistics.write().await;
        stats.hits += 1;
        stats.total_queries += 1;
        stats.hit_rate = stats.hits as f64 / stats.total_queries as f64;
        stats.time_saved += Duration::from_millis(entry.result.computation_time_ms);

        // Update average similarity
        let total_similarity = stats.avg_similarity * (stats.hits - 1) as f32 + similarity;
        stats.avg_similarity = total_similarity / stats.hits as f32;
    }

    async fn record_miss(&self) {
        if !self.config.track_statistics {
            return;
        }

        let mut stats = self.statistics.write().await;
        stats.misses += 1;
        stats.total_queries += 1;
        stats.hit_rate = stats.hits as f64 / stats.total_queries as f64;
    }

    async fn update_size(&self, size: usize) {
        if !self.config.track_statistics {
            return;
        }

        let mut stats = self.statistics.write().await;
        stats.current_size = size;
    }

    fn compute_embedding(&self, _query: &str) -> Result<Array1<f32>> {
        // Simplified - would use actual embedding model
        // For now, return a dummy embedding
        Ok(Array1::zeros(384))
    }

    fn compute_similarity(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> Result<f32> {
        // Use cosine similarity
        let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|a| a * a).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }

    async fn evict(&self, entries: &mut HashMap<String, CacheEntry>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let key_to_remove = match self.config.eviction_strategy {
            EvictionStrategy::LRU => {
                // Remove least recently used
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_access)
                    .map(|(key, _)| key.clone())
            }
            EvictionStrategy::LFU => {
                // Remove least frequently used
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.hit_count)
                    .map(|(key, _)| key.clone())
            }
            EvictionStrategy::TTL => {
                // Remove oldest entry
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.timestamp)
                    .map(|(key, _)| key.clone())
            }
            EvictionStrategy::FIFO => {
                // Remove first inserted (oldest timestamp)
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.timestamp)
                    .map(|(key, _)| key.clone())
            }
        };

        if let Some(key) = key_to_remove {
            entries.remove(&key);

            if self.config.track_statistics {
                let mut stats = self.statistics.write().await;
                stats.evictions += 1;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_match_cache() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());

        let result = CachedResult {
            data: "test data".to_string(),
            metadata: HashMap::new(),
            computation_time_ms: 100,
        };

        cache
            .put("test query".to_string(), result.clone(), None)
            .await
            .unwrap();

        let cached = cache.get("test query").await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().data, "test data");
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());

        let cached = cache.get("nonexistent query").await.unwrap();
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());

        let result = CachedResult {
            data: "test data".to_string(),
            metadata: HashMap::new(),
            computation_time_ms: 100,
        };

        cache
            .put("test query".to_string(), result, None)
            .await
            .unwrap();
        assert!(cache.get("test query").await.unwrap().is_some());

        cache.invalidate("test query").await.unwrap();
        assert!(cache.get("test query").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());

        let result = CachedResult {
            data: "test data".to_string(),
            metadata: HashMap::new(),
            computation_time_ms: 100,
        };

        cache
            .put("test query".to_string(), result, None)
            .await
            .unwrap();
        cache.get("test query").await.unwrap(); // Hit
        cache.get("other query").await.unwrap(); // Miss

        let stats = cache.statistics().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let config = SemanticCacheConfig {
            max_size: 2,
            ..Default::default()
        };

        let cache = SemanticCache::new(config);

        let result = CachedResult {
            data: "test".to_string(),
            metadata: HashMap::new(),
            computation_time_ms: 100,
        };

        cache
            .put("query1".to_string(), result.clone(), None)
            .await
            .unwrap();
        cache
            .put("query2".to_string(), result.clone(), None)
            .await
            .unwrap();
        cache
            .put("query3".to_string(), result.clone(), None)
            .await
            .unwrap(); // Should evict one

        let stats = cache.statistics().await;
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.evictions, 1);
    }
}
