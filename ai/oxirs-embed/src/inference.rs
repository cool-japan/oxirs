//! High-performance inference engine for embedding models

use crate::{EmbeddingModel, ModelStats};
use anyhow::Result;
use oxirs_vec::Vector;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Semaphore;
use tracing::{debug, info};

/// High-performance inference engine with caching and batching
pub struct InferenceEngine {
    model: Arc<RwLock<Box<dyn EmbeddingModel>>>,
    cache: Arc<RwLock<InferenceCache>>,
    config: InferenceConfig,
    batch_processor: BatchProcessor,
}

/// Configuration for inference engine
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum cache size
    pub cache_size: usize,
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Maximum number of concurrent requests
    pub max_concurrent: usize,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    /// Enable result caching
    pub enable_caching: bool,
    /// Warm up cache on startup
    pub warm_up_cache: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            batch_size: 100,
            max_concurrent: 10,
            cache_ttl: 3600, // 1 hour
            enable_caching: true,
            warm_up_cache: false,
        }
    }
}

/// Embedding cache with TTL support
#[derive(Debug)]
pub struct InferenceCache {
    entity_cache: HashMap<String, CacheEntry<Vector>>,
    relation_cache: HashMap<String, CacheEntry<Vector>>,
    triple_score_cache: HashMap<String, CacheEntry<f64>>,
    max_size: usize,
    ttl_seconds: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    timestamp: std::time::SystemTime,
}

impl<T> CacheEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    fn is_expired(&self, ttl_seconds: u64) -> bool {
        if let Ok(elapsed) = self.timestamp.elapsed() {
            elapsed.as_secs() > ttl_seconds
        } else {
            true // If we can't determine elapsed time, consider expired
        }
    }
}

impl InferenceCache {
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            entity_cache: HashMap::new(),
            relation_cache: HashMap::new(),
            triple_score_cache: HashMap::new(),
            max_size,
            ttl_seconds,
        }
    }
    
    pub fn get_entity_embedding(&mut self, entity: &str) -> Option<&Vector> {
        if let Some(entry) = self.entity_cache.get(entity) {
            if !entry.is_expired(self.ttl_seconds) {
                return Some(&entry.value);
            } else {
                self.entity_cache.remove(entity);
            }
        }
        None
    }
    
    pub fn cache_entity_embedding(&mut self, entity: String, embedding: Vector) {
        if self.entity_cache.len() >= self.max_size {
            // Simple LRU: remove oldest entry
            if let Some(oldest_key) = self.find_oldest_entity() {
                self.entity_cache.remove(&oldest_key);
            }
        }
        
        self.entity_cache.insert(entity, CacheEntry::new(embedding));
    }
    
    pub fn get_relation_embedding(&mut self, relation: &str) -> Option<&Vector> {
        if let Some(entry) = self.relation_cache.get(relation) {
            if !entry.is_expired(self.ttl_seconds) {
                return Some(&entry.value);
            } else {
                self.relation_cache.remove(relation);
            }
        }
        None
    }
    
    pub fn cache_relation_embedding(&mut self, relation: String, embedding: Vector) {
        if self.relation_cache.len() >= self.max_size {
            if let Some(oldest_key) = self.find_oldest_relation() {
                self.relation_cache.remove(&oldest_key);
            }
        }
        
        self.relation_cache.insert(relation, CacheEntry::new(embedding));
    }
    
    pub fn get_triple_score(&mut self, key: &str) -> Option<f64> {
        if let Some(entry) = self.triple_score_cache.get(key) {
            if !entry.is_expired(self.ttl_seconds) {
                return Some(entry.value);
            } else {
                self.triple_score_cache.remove(key);
            }
        }
        None
    }
    
    pub fn cache_triple_score(&mut self, key: String, score: f64) {
        if self.triple_score_cache.len() >= self.max_size {
            if let Some(oldest_key) = self.find_oldest_triple() {
                self.triple_score_cache.remove(&oldest_key);
            }
        }
        
        self.triple_score_cache.insert(key, CacheEntry::new(score));
    }
    
    fn find_oldest_entity(&self) -> Option<String> {
        self.entity_cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
    }
    
    fn find_oldest_relation(&self) -> Option<String> {
        self.relation_cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
    }
    
    fn find_oldest_triple(&self) -> Option<String> {
        self.triple_score_cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
    }
    
    pub fn clear(&mut self) {
        self.entity_cache.clear();
        self.relation_cache.clear();
        self.triple_score_cache.clear();
    }
    
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entity_cache_size: self.entity_cache.len(),
            relation_cache_size: self.relation_cache.len(),
            triple_cache_size: self.triple_score_cache.len(),
            max_size: self.max_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entity_cache_size: usize,
    pub relation_cache_size: usize,
    pub triple_cache_size: usize,
    pub max_size: usize,
}

/// Batch processor for efficient bulk operations
#[derive(Debug)]
pub struct BatchProcessor {
    batch_size: usize,
    semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    pub fn new(batch_size: usize, max_concurrent: usize) -> Self {
        Self {
            batch_size,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
    
    pub async fn process_entity_batch(
        &self,
        model: Arc<RwLock<Box<dyn EmbeddingModel>>>,
        entities: Vec<String>,
    ) -> Result<Vec<(String, Result<Vector>)>> {
        let _permit = self.semaphore.acquire().await.unwrap();
        
        let mut results = Vec::new();
        
        for chunk in entities.chunks(self.batch_size) {
            let model_guard = model.read().unwrap();
            for entity in chunk {
                let result = model_guard.get_entity_embedding(entity);
                results.push((entity.clone(), result));
            }
        }
        
        Ok(results)
    }
    
    pub async fn process_relation_batch(
        &self,
        model: Arc<RwLock<Box<dyn EmbeddingModel>>>,
        relations: Vec<String>,
    ) -> Result<Vec<(String, Result<Vector>)>> {
        let _permit = self.semaphore.acquire().await.unwrap();
        
        let mut results = Vec::new();
        
        for chunk in relations.chunks(self.batch_size) {
            let model_guard = model.read().unwrap();
            for relation in chunk {
                let result = model_guard.get_relation_embedding(relation);
                results.push((relation.clone(), result));
            }
        }
        
        Ok(results)
    }
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(
        model: Box<dyn EmbeddingModel>,
        config: InferenceConfig,
    ) -> Self {
        let cache = Arc::new(RwLock::new(InferenceCache::new(
            config.cache_size,
            config.cache_ttl,
        )));
        
        let batch_processor = BatchProcessor::new(
            config.batch_size,
            config.max_concurrent,
        );
        
        Self {
            model: Arc::new(RwLock::new(model)),
            cache,
            config,
            batch_processor,
        }
    }
    
    /// Get entity embedding with caching
    pub async fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        // Check cache first
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                if let Some(cached) = cache.get_entity_embedding(entity) {
                    debug!("Cache hit for entity: {}", entity);
                    return Ok(cached.clone());
                }
            }
        }
        
        // Get from model
        let embedding = {
            let model_guard = self.model.read().unwrap();
            model_guard.get_entity_embedding(entity)?
        };
        
        // Cache result
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                cache.cache_entity_embedding(entity.to_string(), embedding.clone());
            }
        }
        
        Ok(embedding)
    }
    
    /// Get relation embedding with caching
    pub async fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        // Check cache first
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                if let Some(cached) = cache.get_relation_embedding(relation) {
                    debug!("Cache hit for relation: {}", relation);
                    return Ok(cached.clone());
                }
            }
        }
        
        // Get from model
        let embedding = {
            let model_guard = self.model.read().unwrap();
            model_guard.get_relation_embedding(relation)?
        };
        
        // Cache result
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                cache.cache_relation_embedding(relation.to_string(), embedding.clone());
            }
        }
        
        Ok(embedding)
    }
    
    /// Score triple with caching
    pub async fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let cache_key = format!("{}|{}|{}", subject, predicate, object);
        
        // Check cache first
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                if let Some(cached_score) = cache.get_triple_score(&cache_key) {
                    debug!("Cache hit for triple: {}", cache_key);
                    return Ok(cached_score);
                }
            }
        }
        
        // Get from model
        let score = {
            let model_guard = self.model.read().unwrap();
            model_guard.score_triple(subject, predicate, object)?
        };
        
        // Cache result
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                cache.cache_triple_score(cache_key, score);
            }
        }
        
        Ok(score)
    }
    
    /// Batch process entity embeddings
    pub async fn get_entity_embeddings_batch(&self, entities: Vec<String>) -> Result<Vec<(String, Result<Vector>)>> {
        self.batch_processor
            .process_entity_batch(self.model.clone(), entities)
            .await
    }
    
    /// Batch process relation embeddings
    pub async fn get_relation_embeddings_batch(&self, relations: Vec<String>) -> Result<Vec<(String, Result<Vector>)>> {
        self.batch_processor
            .process_relation_batch(self.model.clone(), relations)
            .await
    }
    
    /// Warm up cache with common entities and relations
    pub async fn warm_up_cache(&self) -> Result<()> {
        if !self.config.warm_up_cache {
            return Ok(());
        }
        
        info!("Warming up inference cache...");
        
        let (entities, relations) = {
            let model_guard = self.model.read().unwrap();
            (model_guard.get_entities(), model_guard.get_relations())
        };
        
        // Warm up entity cache
        for entity in entities.iter().take(self.config.cache_size / 2) {
            let _ = self.get_entity_embedding(entity).await;
        }
        
        // Warm up relation cache
        for relation in relations.iter().take(self.config.cache_size / 2) {
            let _ = self.get_relation_embedding(relation).await;
        }
        
        info!("Cache warm-up completed");
        Ok(())
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        let cache_guard = self.cache.read().unwrap();
        Ok(cache_guard.stats())
    }
    
    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache_guard = self.cache.write().unwrap();
        cache_guard.clear();
        info!("Inference cache cleared");
        Ok(())
    }
    
    /// Get model statistics
    pub fn model_stats(&self) -> Result<ModelStats> {
        let model_guard = self.model.read().unwrap();
        Ok(model_guard.get_stats())
    }
}

/// Performance monitoring for inference
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_latency_ms: f64,
    pub throughput_per_second: f64,
}

impl InferenceMetrics {
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.cache_hits as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TransE;
    use crate::ModelConfig;
    
    #[tokio::test]
    async fn test_inference_cache() {
        let mut cache = InferenceCache::new(2, 3600);
        
        let vec1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vec2 = Vector::new(vec![4.0, 5.0, 6.0]);
        
        cache.cache_entity_embedding("entity1".to_string(), vec1.clone());
        cache.cache_entity_embedding("entity2".to_string(), vec2.clone());
        
        assert!(cache.get_entity_embedding("entity1").is_some());
        assert!(cache.get_entity_embedding("entity2").is_some());
        
        // Adding third should evict first (LRU)
        let vec3 = Vector::new(vec![7.0, 8.0, 9.0]);
        cache.cache_entity_embedding("entity3".to_string(), vec3);
        
        assert_eq!(cache.entity_cache.len(), 2);
    }
    
    #[tokio::test]
    async fn test_inference_engine() -> Result<()> {
        let config = ModelConfig::default().with_dimensions(10).with_seed(42);
        let model = TransE::new(config);
        
        let inference_config = InferenceConfig {
            cache_size: 100,
            enable_caching: true,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(Box::new(model), inference_config);
        
        // Test should work even with untrained model
        let stats = engine.model_stats()?;
        assert_eq!(stats.dimensions, 10);
        
        Ok(())
    }
}