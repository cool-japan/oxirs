//! Embedding management for the RAG system
//!
//! Handles embedding generation, caching, and optimization for
//! semantic search and vector similarity operations.

use anyhow::Result;
use async_trait::async_trait;
use oxirs_embed::{EmbeddingModel, Vector as EmbedVector};
use oxirs_vec::{
    embeddings::{EmbeddableContent, EmbeddingStrategy},
    Vector,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Embedding manager for RAG operations
pub struct EmbeddingManager {
    /// Model registry for different embedding models
    model_registry: HashMap<String, Arc<dyn EmbeddingModel>>,
    /// Embedding cache for frequently used embeddings
    embedding_cache: Arc<RwLock<HashMap<String, CachedEmbedding>>>,
    /// Configuration for embedding operations
    config: EmbeddingConfig,
    /// Statistics tracking
    stats: Arc<RwLock<EmbeddingStats>>,
}

impl EmbeddingManager {
    /// Create a new embedding manager
    pub fn new() -> Self {
        Self {
            model_registry: HashMap::new(),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            config: EmbeddingConfig::default(),
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        }
    }

    /// Register an embedding model
    pub fn register_model(&mut self, name: String, model: Arc<dyn EmbeddingModel>) {
        self.model_registry.insert(name, model);
    }

    /// Get embedding for text using the specified model
    pub async fn get_embedding(&self, text: &str, model_name: Option<&str>) -> Result<Vector> {
        let model_key = model_name.unwrap_or(&self.config.default_model);
        let cache_key = format!("{}:{}", model_key, text);

        // Check cache first
        if self.config.enable_caching {
            let cache = self.embedding_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                if !cached.is_expired() {
                    self.update_stats(|stats| stats.cache_hits += 1).await;
                    debug!("Cache hit for embedding: {}", text);
                    return Ok(cached.vector.clone());
                }
            }
        }

        // Cache miss - generate embedding
        let model = self
            .model_registry
            .get(model_key)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_key))?;

        let embed_vector = model.get_entity_embedding(text)?;
        let vector = Vector::new(embed_vector.values);

        // Cache the result
        if self.config.enable_caching {
            let cached_embedding = CachedEmbedding::new(vector.clone());
            let mut cache = self.embedding_cache.write().await;
            cache.insert(cache_key, cached_embedding);

            // Cleanup old entries if cache is too large
            if cache.len() > self.config.max_cache_size {
                self.cleanup_cache(&mut cache).await;
            }
        }

        self.update_stats(|stats| {
            stats.cache_misses += 1;
            stats.embeddings_generated += 1;
        })
        .await;

        info!("Generated embedding for text: {}", text);
        Ok(vector)
    }

    /// Get embeddings for multiple texts in batch
    pub async fn get_embeddings_batch(
        &self,
        texts: &[String],
        model_name: Option<&str>,
    ) -> Result<Vec<Vector>> {
        let mut embeddings = Vec::new();

        // Process in batches for efficiency
        for batch in texts.chunks(self.config.batch_size) {
            let batch_results = self.process_batch(batch, model_name).await?;
            embeddings.extend(batch_results);
        }

        Ok(embeddings)
    }

    /// Precompute and cache embeddings for common terms
    pub async fn precompute_embeddings(
        &self,
        terms: &[String],
        model_name: Option<&str>,
    ) -> Result<()> {
        info!("Precomputing embeddings for {} terms", terms.len());

        for term in terms {
            if let Err(e) = self.get_embedding(term, model_name).await {
                warn!("Failed to precompute embedding for '{}': {}", term, e);
            }
        }

        Ok(())
    }

    /// Clear the embedding cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.embedding_cache.write().await;
        {
            cache.clear();
            self.update_stats(|stats| stats.cache_cleared += 1).await;
            info!("Embedding cache cleared");
        }
        Ok(())
    }

    /// Get embedding statistics
    pub async fn get_stats(&self) -> EmbeddingStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EmbeddingConfig) {
        self.config = config;
    }

    /// Process a batch of texts
    async fn process_batch(
        &self,
        texts: &[String],
        model_name: Option<&str>,
    ) -> Result<Vec<Vector>> {
        let mut results = Vec::new();

        for text in texts {
            let embedding = self.get_embedding(text, model_name).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Cleanup old cache entries
    async fn cleanup_cache(&self, cache: &mut HashMap<String, CachedEmbedding>) {
        // Remove expired entries first
        cache.retain(|_, cached| !cached.is_expired());

        // If still too large, remove oldest entries
        if cache.len() > self.config.max_cache_size {
            let entries: Vec<_> = cache
                .iter()
                .map(|(k, v)| (k.clone(), v.created_at))
                .collect();
            let mut sorted_entries = entries;
            sorted_entries.sort_by_key(|(_, created_at)| *created_at);

            let to_remove = cache.len() - self.config.max_cache_size;
            for (key, _) in sorted_entries.iter().take(to_remove) {
                cache.remove(key);
            }
        }
    }

    /// Update statistics
    async fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut EmbeddingStats),
    {
        let mut stats = self.stats.write().await;
        {
            updater(&mut stats);
        }
    }
}

/// Cached embedding with metadata
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub vector: Vector,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
    pub ttl: chrono::Duration,
}

impl CachedEmbedding {
    pub fn new(vector: Vector) -> Self {
        Self {
            vector,
            created_at: chrono::Utc::now(),
            access_count: 1,
            ttl: chrono::Duration::hours(24), // Default 24 hour TTL
        }
    }

    /// Check if the cached embedding has expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.created_at + self.ttl
    }

    /// Update access count and timestamp
    pub fn touch(&mut self) {
        self.access_count += 1;
    }
}

/// Configuration for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Default model to use for embeddings
    pub default_model: String,
    /// Enable embedding caching
    pub enable_caching: bool,
    /// Maximum number of cached embeddings
    pub max_cache_size: usize,
    /// Batch size for processing multiple texts
    pub batch_size: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Number of worker threads for parallel processing
    pub num_workers: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            default_model: "default".to_string(),
            enable_caching: true,
            max_cache_size: 10000,
            batch_size: 100,
            enable_parallel: true,
            num_workers: 4,
        }
    }
}

/// Statistics for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of embeddings generated
    pub embeddings_generated: u64,
    /// Number of times cache was cleared
    pub cache_cleared: u64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: u64,
    /// Average processing time per embedding in milliseconds
    pub avg_processing_time_ms: f64,
}

impl Default for EmbeddingStats {
    fn default() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            embeddings_generated: 0,
            cache_cleared: 0,
            total_processing_time_ms: 0,
            avg_processing_time_ms: 0.0,
        }
    }
}

impl EmbeddingStats {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            self.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Update average processing time
    pub fn update_avg_processing_time(&mut self) {
        if self.embeddings_generated > 0 {
            self.avg_processing_time_ms =
                self.total_processing_time_ms as f64 / self.embeddings_generated as f64;
        }
    }
}

/// Simple embedding model wrapper for testing
pub struct SimpleEmbeddingModel {
    dimensions: usize,
    config: oxirs_embed::ModelConfig,
    model_id: uuid::Uuid,
}

impl SimpleEmbeddingModel {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            config: oxirs_embed::ModelConfig {
                dimensions,
                learning_rate: 0.01,
                l2_reg: 0.001,
                max_epochs: 100,
                batch_size: 128,
                negative_samples: 5,
                seed: Some(42),
                use_gpu: false,
                model_params: std::collections::HashMap::new(),
            },
            model_id: uuid::Uuid::new_v4(),
        }
    }

    /// Convenience method for single string embedding
    pub fn embed(&self, text: &str) -> Result<Vector> {
        let embed_vector = self.get_entity_embedding(text)?;
        // Convert oxirs_embed::Vector to oxirs_vec::Vector
        Ok(Vector::new(embed_vector.values))
    }
}

#[async_trait]
impl EmbeddingModel for SimpleEmbeddingModel {
    fn config(&self) -> &oxirs_embed::ModelConfig {
        &self.config
    }

    fn model_id(&self) -> &uuid::Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "simple"
    }

    fn add_triple(&mut self, _triple: oxirs_embed::Triple) -> Result<()> {
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<oxirs_embed::TrainingStats> {
        let epochs = epochs.unwrap_or(self.config.max_epochs);

        // Simple mock training for testing purposes
        let mut loss_history = Vec::new();
        for i in 0..epochs {
            loss_history.push(0.1 * (epochs - i) as f64 / epochs as f64);
        }

        let stats = oxirs_embed::TrainingStats {
            epochs_completed: epochs,
            convergence_achieved: true,
            loss_history,
            final_loss: 0.1,
            training_time_seconds: 1.0,
        };

        Ok(stats)
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<EmbedVector> {
        // Simple hash-based embedding for testing
        let mut embedding = vec![0.0f32; self.dimensions];
        let bytes = entity.as_bytes();

        for (i, &byte) in bytes.iter().enumerate() {
            let idx = i % self.dimensions;
            embedding[idx] += (byte as f32) / 255.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(EmbedVector::new(embedding))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<EmbedVector> {
        self.get_entity_embedding(relation)
    }

    fn score_triple(&self, _subject: &str, _predicate: &str, _object: &str) -> Result<f64> {
        Ok(0.5) // Placeholder
    }

    fn predict_objects(
        &self,
        _subject: &str,
        _predicate: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        Ok(vec![])
    }

    fn predict_subjects(
        &self,
        _predicate: &str,
        _object: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        Ok(vec![])
    }

    fn predict_relations(
        &self,
        _subject: &str,
        _object: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        Ok(vec![])
    }

    fn get_entities(&self) -> Vec<String> {
        vec![]
    }

    fn get_relations(&self) -> Vec<String> {
        vec![]
    }

    fn get_stats(&self) -> oxirs_embed::ModelStats {
        oxirs_embed::ModelStats::default()
    }

    fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn clear(&mut self) {
        // Nothing to clear
    }

    fn is_trained(&self) -> bool {
        true // Always "trained" for simple model
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            let embedding = self.get_entity_embedding(text)?;
            results.push(embedding.values);
        }
        Ok(results)
    }
}

impl Default for EmbeddingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_embedding_manager_creation() {
        let manager = EmbeddingManager::new();
        assert!(manager.model_registry.is_empty());
    }

    #[tokio::test]
    async fn test_simple_embedding_model() {
        let model = SimpleEmbeddingModel::new(128);
        let embedding = model.get_entity_embedding("test").unwrap();
        assert_eq!(embedding.values.len(), 128);
    }

    #[tokio::test]
    async fn test_embedding_caching() {
        let mut manager = EmbeddingManager::new();
        let model = Arc::new(SimpleEmbeddingModel::new(128));
        manager.register_model("test_model".to_string(), model);

        // First call should miss cache
        let embedding1 = manager
            .get_embedding("test", Some("test_model"))
            .await
            .unwrap();

        // Second call should hit cache
        let embedding2 = manager
            .get_embedding("test", Some("test_model"))
            .await
            .unwrap();

        assert_eq!(embedding1.values, embedding2.values);

        let stats = manager.get_stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[tokio::test]
    async fn test_batch_embedding() {
        let mut manager = EmbeddingManager::new();
        let model = Arc::new(SimpleEmbeddingModel::new(128));
        manager.register_model("test_model".to_string(), model);

        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];
        let embeddings = manager
            .get_embeddings_batch(&texts, Some("test_model"))
            .await
            .unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 128);
        }
    }

    #[test]
    fn test_cached_embedding_expiry() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);
        let mut cached = CachedEmbedding::new(vector);

        // Should not be expired immediately
        assert!(!cached.is_expired());

        // Set TTL to very short duration for testing
        cached.ttl = chrono::Duration::milliseconds(1);
        std::thread::sleep(std::time::Duration::from_millis(2));

        // Should now be expired
        assert!(cached.is_expired());
    }

    #[test]
    fn test_embedding_stats() {
        let mut stats = EmbeddingStats::default();
        stats.cache_hits = 8;
        stats.cache_misses = 2;

        assert_eq!(stats.cache_hit_ratio(), 0.8);

        stats.embeddings_generated = 5;
        stats.total_processing_time_ms = 1000;
        stats.update_avg_processing_time();

        assert_eq!(stats.avg_processing_time_ms, 200.0);
    }
}
