//! Advanced caching and precomputation system for embedding models
//!
//! This module provides multi-level caching for embeddings, computation results,
//! and intelligent precomputation strategies for improved performance.

use crate::{EmbeddingModel, Vector};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
// Removed unused imports
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Type alias for similarity cache
type SimilarityCache = Arc<RwLock<LRUCache<String, Vec<(String, f64)>>>>;

/// Multi-level caching system for embeddings and computations
pub struct CacheManager {
    /// L1 Cache: Hot embeddings (fastest access)
    l1_cache: Arc<RwLock<LRUCache<String, CachedEmbedding>>>,
    /// L2 Cache: Computation results (intermediate speed)
    l2_cache: Arc<RwLock<LRUCache<ComputationKey, CachedComputation>>>,
    /// L3 Cache: Similarity cache (bulk operations)
    l3_cache: SimilarityCache,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Background cleanup task
    cleanup_task: Option<JoinHandle<()>>,
    /// Cache warming strategy
    #[allow(dead_code)]
    warming_strategy: WarmingStrategy,
}

/// Configuration for the caching system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size (number of embeddings)
    pub l1_max_size: usize,
    /// L2 cache size (number of computation results)
    pub l2_max_size: usize,
    /// L3 cache size (number of similarity results)
    pub l3_max_size: usize,
    /// Cache entry TTL in seconds
    pub ttl_seconds: u64,
    /// Enable cache warming
    pub enable_warming: bool,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Background cleanup interval
    pub cleanup_interval_seconds: u64,
    /// Enable cache compression
    pub enable_compression: bool,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 10_000,
            l2_max_size: 50_000,
            l3_max_size: 100_000,
            ttl_seconds: 3600, // 1 hour
            enable_warming: true,
            eviction_policy: EvictionPolicy::LRU,
            cleanup_interval_seconds: 300, // 5 minutes
            enable_compression: true,
            max_memory_mb: 1024, // 1GB
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Adaptive,
}

/// Cache warming strategies
#[derive(Debug, Clone)]
pub enum WarmingStrategy {
    /// Pre-populate with most frequently accessed entities
    MostFrequent(usize),
    /// Pre-populate with entities from recent queries
    RecentQueries(usize),
    /// Pre-populate with entities based on graph centrality
    GraphCentrality(usize),
    /// No warming
    None,
}

impl Default for WarmingStrategy {
    fn default() -> Self {
        WarmingStrategy::MostFrequent(1000)
    }
}

/// Cached embedding with metadata
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    /// The embedding vector
    pub embedding: Vector,
    /// When this was cached
    pub cached_at: DateTime<Utc>,
    /// Last access time
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether this is compressed
    pub is_compressed: bool,
}

/// Key for computation caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputationKey {
    pub operation: String,
    pub inputs: Vec<String>,
    pub model_id: Uuid,
}

/// Cached computation result
#[derive(Debug, Clone)]
pub struct CachedComputation {
    /// The computation result
    pub result: ComputationResult,
    /// When this was cached
    pub cached_at: DateTime<Utc>,
    /// Last access time
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Computation time saved (in microseconds)
    pub time_saved_us: u64,
}

/// Types of computation results that can be cached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationResult {
    TripleScore(f64),
    EntitySimilarity(Vec<(String, f64)>),
    PredictionResults(Vec<(String, f64)>),
    AttentionWeights(Vec<f64>),
    IntermediateActivations(Vec<f64>),
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Total memory usage in bytes
    pub memory_usage_bytes: usize,
    /// L1 cache stats
    pub l1_stats: LevelStats,
    /// L2 cache stats
    pub l2_stats: LevelStats,
    /// L3 cache stats
    pub l3_stats: LevelStats,
    /// Time saved by caching (in seconds)
    pub total_time_saved_seconds: f64,
}

/// Statistics for a cache level
#[derive(Debug, Clone)]
pub struct LevelStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub capacity: usize,
    pub memory_bytes: usize,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            hit_rate: 0.0,
            memory_usage_bytes: 0,
            l1_stats: LevelStats {
                hits: 0,
                misses: 0,
                size: 0,
                capacity: 0,
                memory_bytes: 0,
            },
            l2_stats: LevelStats {
                hits: 0,
                misses: 0,
                size: 0,
                capacity: 0,
                memory_bytes: 0,
            },
            l3_stats: LevelStats {
                hits: 0,
                misses: 0,
                size: 0,
                capacity: 0,
                memory_bytes: 0,
            },
            total_time_saved_seconds: 0.0,
        }
    }
}

/// LRU Cache implementation
pub struct LRUCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    capacity: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
    access_times: HashMap<K, Instant>,
    ttl: Duration,
}

impl<K, V> LRUCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            order: VecDeque::new(),
            access_times: HashMap::new(),
            ttl,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        // Check TTL
        if let Some(access_time) = self.access_times.get(key) {
            if access_time.elapsed() > self.ttl {
                self.remove(key);
                return None;
            }
        }

        if let Some(value) = self.map.get(key).cloned() {
            // Move to front
            self.move_to_front(key);
            self.access_times.insert(key.clone(), Instant::now());
            Some(value)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            // Update existing
            self.map.insert(key.clone(), value);
            self.move_to_front(&key);
        } else {
            // Add new
            if self.map.len() >= self.capacity {
                self.evict_lru();
            }
            self.map.insert(key.clone(), value);
            self.order.push_front(key.clone());
        }
        self.access_times.insert(key, Instant::now());
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.map.remove(key) {
            self.order.retain(|k| k != key);
            self.access_times.remove(key);
            Some(value)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.access_times.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn move_to_front(&mut self, key: &K) {
        self.order.retain(|k| k != key);
        self.order.push_front(key.clone());
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.order.pop_back() {
            self.map.remove(&key);
            self.access_times.remove(&key);
        }
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&mut self) -> usize {
        let now = Instant::now();
        let mut expired_keys = Vec::new();

        for (key, access_time) in &self.access_times {
            if now.duration_since(*access_time) > self.ttl {
                expired_keys.push(key.clone());
            }
        }

        let count = expired_keys.len();
        for key in expired_keys {
            self.remove(&key);
        }

        count
    }
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: CacheConfig) -> Self {
        let ttl = Duration::from_secs(config.ttl_seconds);

        Self {
            l1_cache: Arc::new(RwLock::new(LRUCache::new(config.l1_max_size, ttl))),
            l2_cache: Arc::new(RwLock::new(LRUCache::new(config.l2_max_size, ttl))),
            l3_cache: Arc::new(RwLock::new(LRUCache::new(config.l3_max_size, ttl))),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            cleanup_task: None,
            warming_strategy: WarmingStrategy::default(),
        }
    }

    /// Start the cache manager with background tasks
    pub async fn start(&mut self) -> Result<()> {
        // Start cleanup task
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);
        let l1_cache = Arc::clone(&self.l1_cache);
        let l2_cache = Arc::clone(&self.l2_cache);
        let l3_cache = Arc::clone(&self.l3_cache);
        let stats = Arc::clone(&self.stats);

        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                interval.tick().await;

                // Cleanup expired entries
                let expired_l1 = {
                    let mut cache = l1_cache.write().unwrap();
                    cache.cleanup_expired()
                };

                let expired_l2 = {
                    let mut cache = l2_cache.write().unwrap();
                    cache.cleanup_expired()
                };

                let expired_l3 = {
                    let mut cache = l3_cache.write().unwrap();
                    cache.cleanup_expired()
                };

                let total_expired = expired_l1 + expired_l2 + expired_l3;
                if total_expired > 0 {
                    debug!("Cleaned up {} expired cache entries", total_expired);
                }

                // Update stats
                {
                    let mut stats = stats.write().unwrap();
                    stats.l1_stats.size = l1_cache.read().unwrap().len();
                    stats.l2_stats.size = l2_cache.read().unwrap().len();
                    stats.l3_stats.size = l3_cache.read().unwrap().len();

                    // Update hit rate
                    let total_requests = stats.total_hits + stats.total_misses;
                    if total_requests > 0 {
                        stats.hit_rate = stats.total_hits as f64 / total_requests as f64;
                    }
                }
            }
        });

        self.cleanup_task = Some(cleanup_task);
        info!(
            "Cache manager started with cleanup interval: {:?}",
            cleanup_interval
        );
        Ok(())
    }

    /// Stop the cache manager
    pub async fn stop(&mut self) {
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
            info!("Cache manager stopped");
        }
    }

    /// Get cached embedding
    pub fn get_embedding(&self, entity: &str) -> Option<Vector> {
        let start = Instant::now();

        let result = {
            let mut cache = self.l1_cache.write().unwrap();
            cache.get(&entity.to_string())
        };

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            if result.is_some() {
                stats.total_hits += 1;
                stats.l1_stats.hits += 1;
                let time_saved = start.elapsed().as_micros() as f64 / 1_000_000.0;
                stats.total_time_saved_seconds += time_saved;
            } else {
                stats.total_misses += 1;
                stats.l1_stats.misses += 1;
            }
        }

        result.map(|cached| {
            // Update access info
            let mut cached = cached;
            cached.last_accessed = Utc::now();
            cached.access_count += 1;
            cached.embedding
        })
    }

    /// Cache an embedding
    pub fn put_embedding(&self, entity: String, embedding: Vector) {
        let cached = CachedEmbedding {
            size_bytes: embedding.values.len() * std::mem::size_of::<f32>(),
            embedding,
            cached_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            is_compressed: false,
        };

        {
            let mut cache = self.l1_cache.write().unwrap();
            cache.put(entity, cached);
        }

        // Update capacity stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.l1_stats.capacity = self.config.l1_max_size;
        }
    }

    /// Get cached computation result
    pub fn get_computation(&self, key: &ComputationKey) -> Option<ComputationResult> {
        let start = Instant::now();

        let result = {
            let mut cache = self.l2_cache.write().unwrap();
            cache.get(key)
        };

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            if result.is_some() {
                stats.total_hits += 1;
                stats.l2_stats.hits += 1;
                let time_saved = start.elapsed().as_micros() as f64 / 1_000_000.0;
                stats.total_time_saved_seconds += time_saved;
            } else {
                stats.total_misses += 1;
                stats.l2_stats.misses += 1;
            }
        }

        result.map(|cached| cached.result)
    }

    /// Cache a computation result
    pub fn put_computation(
        &self,
        key: ComputationKey,
        result: ComputationResult,
        computation_time_us: u64,
    ) {
        let cached = CachedComputation {
            result,
            cached_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            time_saved_us: computation_time_us,
        };

        {
            let mut cache = self.l2_cache.write().unwrap();
            cache.put(key, cached);
        }
    }

    /// Get cached similarity results
    pub fn get_similarity_cache(&self, query: &str) -> Option<Vec<(String, f64)>> {
        let start = Instant::now();

        let result = {
            let mut cache = self.l3_cache.write().unwrap();
            cache.get(&query.to_string())
        };

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            if result.is_some() {
                stats.total_hits += 1;
                stats.l3_stats.hits += 1;
                let time_saved = start.elapsed().as_micros() as f64 / 1_000_000.0;
                stats.total_time_saved_seconds += time_saved;
            } else {
                stats.total_misses += 1;
                stats.l3_stats.misses += 1;
            }
        }

        result
    }

    /// Cache similarity results
    pub fn put_similarity_cache(&self, query: String, results: Vec<(String, f64)>) {
        let mut cache = self.l3_cache.write().unwrap();
        cache.put(query, results);
    }

    /// Warm up cache with frequently accessed entities
    pub async fn warm_cache(
        &self,
        model: &dyn EmbeddingModel,
        entities: Vec<String>,
    ) -> Result<usize> {
        if !self.config.enable_warming {
            return Ok(0);
        }

        info!("Starting cache warming with {} entities", entities.len());
        let mut warmed_count = 0;

        for entity in entities {
            // Check if already cached
            if self.get_embedding(&entity).is_some() {
                continue;
            }

            // Get embedding and cache it
            match model.get_entity_embedding(&entity) {
                Ok(embedding) => {
                    self.put_embedding(entity, embedding);
                    warmed_count += 1;
                }
                Err(e) => {
                    warn!("Failed to warm cache for entity {}: {}", entity, e);
                }
            }
        }

        info!("Cache warming completed: {} entities cached", warmed_count);
        Ok(warmed_count)
    }

    /// Precompute and cache common operations
    pub async fn precompute_common_operations(
        &self,
        model: &dyn EmbeddingModel,
        common_queries: Vec<(String, String)>,
    ) -> Result<usize> {
        info!(
            "Starting precomputation for {} common queries",
            common_queries.len()
        );
        let mut precomputed_count = 0;

        for (subject, predicate) in common_queries {
            // Precompute object predictions
            let key = ComputationKey {
                operation: "predict_objects".to_string(),
                inputs: vec![subject.clone(), predicate.clone()],
                model_id: *model.model_id(),
            };

            // Check if already cached
            if self.get_computation(&key).is_some() {
                continue;
            }

            let start = Instant::now();
            match model.predict_objects(&subject, &predicate, 10) {
                Ok(predictions) => {
                    let computation_time = start.elapsed().as_micros() as u64;
                    let result = ComputationResult::PredictionResults(predictions);
                    self.put_computation(key, result, computation_time);
                    precomputed_count += 1;
                }
                Err(e) => {
                    warn!(
                        "Failed to precompute prediction for ({}, {}): {}",
                        subject, predicate, e
                    );
                }
            }
        }

        info!(
            "Precomputation completed: {} operations cached",
            precomputed_count
        );
        Ok(precomputed_count)
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        {
            let mut cache = self.l1_cache.write().unwrap();
            cache.clear();
        }
        {
            let mut cache = self.l2_cache.write().unwrap();
            cache.clear();
        }
        {
            let mut cache = self.l3_cache.write().unwrap();
            cache.clear();
        }

        // Reset stats
        {
            let mut stats = self.stats.write().unwrap();
            *stats = CacheStats::default();
        }

        info!("All caches cleared");
    }

    /// Get memory usage estimation
    pub fn estimate_memory_usage(&self) -> usize {
        let l1_size = {
            let cache = self.l1_cache.read().unwrap();
            cache.len() * std::mem::size_of::<CachedEmbedding>()
        };

        let l2_size = {
            let cache = self.l2_cache.read().unwrap();
            cache.len() * std::mem::size_of::<CachedComputation>()
        };

        let l3_size = {
            let cache = self.l3_cache.read().unwrap();
            cache.len() * std::mem::size_of::<Vec<(String, f64)>>()
        };

        l1_size + l2_size + l3_size
    }

    /// Adaptive cache resizing based on usage patterns
    pub fn adaptive_resize(&mut self) {
        let stats = self.get_stats();

        // Resize based on hit rates and memory usage
        if stats.l1_stats.hits > stats.l1_stats.misses * 2
            && stats.memory_usage_bytes < self.config.max_memory_mb * 1024 * 1024 / 2
        {
            // High hit rate and low memory usage - increase L1 cache
            self.config.l1_max_size = (self.config.l1_max_size as f64 * 1.2) as usize;
            info!("Increased L1 cache size to {}", self.config.l1_max_size);
        } else if stats.l1_stats.misses > stats.l1_stats.hits * 2 {
            // High miss rate - decrease L1 cache
            self.config.l1_max_size = (self.config.l1_max_size as f64 * 0.8) as usize;
            info!("Decreased L1 cache size to {}", self.config.l1_max_size);
        }
    }
}

/// Cache-aware embedding wrapper
pub struct CachedEmbeddingModel {
    model: Box<dyn EmbeddingModel>,
    cache_manager: Arc<CacheManager>,
}

impl CachedEmbeddingModel {
    pub fn new(model: Box<dyn EmbeddingModel>, cache_manager: Arc<CacheManager>) -> Self {
        Self {
            model,
            cache_manager,
        }
    }

    /// Get entity embedding with caching
    pub fn get_entity_embedding_cached(&self, entity: &str) -> Result<Vector> {
        // Try cache first
        if let Some(cached) = self.cache_manager.get_embedding(entity) {
            return Ok(cached);
        }

        // Cache miss - get from model
        let embedding = self.model.get_entity_embedding(entity)?;

        // Cache the result
        self.cache_manager
            .put_embedding(entity.to_string(), embedding.clone());

        Ok(embedding)
    }

    /// Score triple with caching
    pub fn score_triple_cached(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let key = ComputationKey {
            operation: "score_triple".to_string(),
            inputs: vec![
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            ],
            model_id: *self.model.model_id(),
        };

        // Try cache first
        if let Some(ComputationResult::TripleScore(score)) =
            self.cache_manager.get_computation(&key)
        {
            return Ok(score);
        }

        // Cache miss - compute from model
        let start = Instant::now();
        let score = self.model.score_triple(subject, predicate, object)?;
        let computation_time = start.elapsed().as_micros() as u64;

        // Cache the result
        self.cache_manager.put_computation(
            key,
            ComputationResult::TripleScore(score),
            computation_time,
        );

        Ok(score)
    }

    /// Predict objects with caching
    pub fn predict_objects_cached(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let key = ComputationKey {
            operation: format!("predict_objects_{}", k),
            inputs: vec![subject.to_string(), predicate.to_string()],
            model_id: *self.model.model_id(),
        };

        // Try cache first
        if let Some(ComputationResult::PredictionResults(predictions)) =
            self.cache_manager.get_computation(&key)
        {
            return Ok(predictions);
        }

        // Cache miss - compute from model
        let start = Instant::now();
        let predictions = self.model.predict_objects(subject, predicate, k)?;
        let computation_time = start.elapsed().as_micros() as u64;

        // Cache the result
        self.cache_manager.put_computation(
            key,
            ComputationResult::PredictionResults(predictions.clone()),
            computation_time,
        );

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LRUCache::new(3, Duration::from_secs(60));

        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.len(), 3);

        // Add one more - should evict least recently used
        cache.put("d".to_string(), 4);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"a".to_string()), None); // Should be evicted
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_max_size, 10_000);
        assert_eq!(config.l2_max_size, 50_000);
        assert_eq!(config.l3_max_size, 100_000);
        assert_eq!(config.ttl_seconds, 3600);
        assert!(config.enable_warming);
    }

    #[tokio::test]
    async fn test_cache_manager_basic() {
        let config = CacheConfig {
            l1_max_size: 100,
            l2_max_size: 100,
            l3_max_size: 100,
            ..Default::default()
        };

        let cache_manager = CacheManager::new(config);

        // Test embedding caching
        let embedding = Vector::new(vec![1.0, 2.0, 3.0]);
        cache_manager.put_embedding("test_entity".to_string(), embedding.clone());

        let cached = cache_manager.get_embedding("test_entity");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().values, embedding.values);

        // Test computation caching
        let key = ComputationKey {
            operation: "test_op".to_string(),
            inputs: vec!["input1".to_string()],
            model_id: Uuid::new_v4(),
        };

        let result = ComputationResult::TripleScore(0.85);
        cache_manager.put_computation(key.clone(), result, 1000);

        let cached_result = cache_manager.get_computation(&key);
        assert!(cached_result.is_some());

        if let Some(ComputationResult::TripleScore(score)) = cached_result {
            assert_eq!(score, 0.85);
        } else {
            panic!("Expected TripleScore result");
        }
    }

    #[test]
    fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache_manager = CacheManager::new(config);

        // Initially empty
        let stats = cache_manager.get_stats();
        assert_eq!(stats.total_hits, 0);
        assert_eq!(stats.total_misses, 0);

        // Cache miss
        let result = cache_manager.get_embedding("nonexistent");
        assert!(result.is_none());

        let stats = cache_manager.get_stats();
        assert_eq!(stats.total_misses, 1);

        // Cache hit
        let embedding = Vector::new(vec![1.0, 2.0, 3.0]);
        cache_manager.put_embedding("test".to_string(), embedding);
        let cached = cache_manager.get_embedding("test");
        assert!(cached.is_some());

        let stats = cache_manager.get_stats();
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn test_computation_key_equality() {
        let key1 = ComputationKey {
            operation: "test".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            model_id: Uuid::new_v4(),
        };

        let key2 = ComputationKey {
            operation: "test".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            model_id: key1.model_id,
        };

        let key3 = ComputationKey {
            operation: "different".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            model_id: key1.model_id,
        };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
