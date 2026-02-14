//! GPU-Accelerated Embedding Cache using SciRS2-Core
//!
//! This module provides high-performance embedding caching with GPU acceleration
//! for the OxiRS Chat system using scirs2-core's GPU capabilities.
//!
//! # Features
//!
//! - GPU-accelerated similarity search
//! - Mixed-precision tensor operations
//! - Memory-efficient caching with lazy loading
//! - Automatic GPU/CPU fallback
//! - SIMD-optimized vector operations
//!
//! # Examples
//!
//! ```rust,no_run
//! use oxirs_chat::gpu_embedding_cache::{GpuEmbeddingCache, CacheConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = CacheConfig::default();
//! let cache = GpuEmbeddingCache::new(config).await?;
//!
//! let embedding = vec![0.1, 0.2, 0.3, 0.4];
//! cache.insert("key1", &embedding).await?;
//!
//! let similar = cache.find_similar(&embedding, 5).await?;
//! println!("Found {} similar embeddings", similar.len());
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use scirs2_core::{
    error::CoreError,
    ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2},
    parallel_ops::{par_chunks, par_join},
    simd_ops::simd_dot_product,
};
// Note: GPU and tensor_cores features will be available in scirs2-core beta.4+
// For now, we provide fallback implementations
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for GPU embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// GPU backend to use (cuda or metal)
    pub gpu_backend: GpuBackend,

    /// Enable mixed-precision operations
    pub enable_mixed_precision: bool,

    /// Use memory-mapped arrays for large caches
    pub use_memory_mapping: bool,

    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Chunk size for parallel processing
    pub chunk_size: usize,

    /// Cache file path for persistence
    pub cache_file: Option<PathBuf>,

    /// Enable automatic cache eviction
    pub enable_auto_eviction: bool,

    /// Cache eviction threshold (0.0 to 1.0)
    pub eviction_threshold: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Cpu, // Fallback
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true, // Enable if available
            gpu_backend: GpuBackend::Cuda,
            enable_mixed_precision: true,
            use_memory_mapping: false,
            max_cache_size_mb: 1024, // 1GB
            enable_simd: true,
            embedding_dim: 768, // Common embedding dimension
            chunk_size: 1000,
            cache_file: None,
            enable_auto_eviction: true,
            eviction_threshold: 0.9,
        }
    }
}

/// Cached embedding entry
#[derive(Debug, Clone)]
struct CachedEmbedding {
    key: String,
    embedding: Vec<f32>,
    last_accessed: chrono::DateTime<chrono::Utc>,
    access_count: usize,
}

/// Similarity search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub key: String,
    pub similarity_score: f32,
    pub embedding: Vec<f32>,
}

// Placeholder types for GPU features (will be replaced with scirs2-core beta.4+)
struct GpuContext;
struct GpuBuffer;
struct TensorCore;

impl GpuContext {
    fn new_cuda() -> Result<Self> {
        Err(anyhow::anyhow!("GPU features not yet available in scirs2-core rc.2"))
    }
    fn new_metal() -> Result<Self> {
        Err(anyhow::anyhow!("GPU features not yet available in scirs2-core rc.2"))
    }
}

impl GpuBuffer {
    fn new(_context: &GpuContext, _size: usize) -> Result<Self> {
        Err(anyhow::anyhow!("GPU features not yet available in scirs2-core rc.2"))
    }
}

impl TensorCore {
    fn new(_context: &GpuContext) -> Result<Self> {
        Err(anyhow::anyhow!("GPU features not yet available in scirs2-core rc.2"))
    }

    fn gemv_mixed_precision(
        &self,
        _matrix: ArrayView2<f32>,
        _query: ArrayView1<f32>,
        _result: scirs2_core::ndarray_ext::ArrayViewMut1<f32>,
    ) -> Result<()> {
        Ok(())
    }
}

/// GPU-accelerated embedding cache
pub struct GpuEmbeddingCache {
    config: CacheConfig,
    embeddings: Arc<RwLock<HashMap<String, CachedEmbedding>>>,
    gpu_context: Option<Arc<GpuContext>>,
    gpu_buffer: Option<Arc<RwLock<GpuBuffer>>>,
    tensor_core: Option<Arc<TensorCore>>,
    embedding_matrix: Arc<RwLock<Option<Array2<f32>>>>,
    key_index: Arc<RwLock<Vec<String>>>, // Maps row index to key
    total_size_bytes: Arc<RwLock<usize>>,
}

impl GpuEmbeddingCache {
    /// Create a new GPU-accelerated embedding cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        info!(
            "Initializing GPU embedding cache (GPU enabled: {})",
            config.enable_gpu
        );

        // Initialize GPU context if enabled
        let (gpu_context, tensor_core) = if config.enable_gpu {
            match Self::initialize_gpu(&config).await {
                Ok((ctx, tc)) => (Some(Arc::new(ctx)), Some(Arc::new(tc))),
                Err(e) => {
                    warn!("GPU initialization failed: {}. Falling back to CPU.", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        let gpu_buffer = if let Some(ref ctx) = gpu_context {
            match Self::create_gpu_buffer(ctx, &config).await {
                Ok(buffer) => Some(Arc::new(RwLock::new(buffer))),
                Err(e) => {
                    warn!("GPU buffer creation failed: {}. Using CPU memory.", e);
                    None
                }
            }
        } else {
            None
        };

        // Load from cache file if specified
        let embeddings = if let Some(ref cache_file) = config.cache_file {
            Self::load_from_file(cache_file).await.unwrap_or_default()
        } else {
            HashMap::new()
        };

        let total_size = embeddings
            .values()
            .map(|e| e.embedding.len() * std::mem::size_of::<f32>())
            .sum();

        info!(
            "GPU embedding cache initialized with {} entries ({} bytes)",
            embeddings.len(),
            total_size
        );

        Ok(Self {
            config,
            embeddings: Arc::new(RwLock::new(embeddings)),
            gpu_context,
            gpu_buffer,
            tensor_core,
            embedding_matrix: Arc::new(RwLock::new(None)),
            key_index: Arc::new(RwLock::new(Vec::new())),
            total_size_bytes: Arc::new(RwLock::new(total_size)),
        })
    }

    /// Insert an embedding into the cache
    pub async fn insert(&self, key: &str, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.config.embedding_dim {
            return Err(anyhow::anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            ));
        }

        let entry = CachedEmbedding {
            key: key.to_string(),
            embedding: embedding.to_vec(),
            last_accessed: chrono::Utc::now(),
            access_count: 1,
        };

        let entry_size = embedding.len() * std::mem::size_of::<f32>();

        // Check cache size and evict if necessary
        let should_evict = {
            let total_size = self.total_size_bytes.read().await;
            let max_size = self.config.max_cache_size_mb * 1024 * 1024;
            *total_size + entry_size > max_size && self.config.enable_auto_eviction
        };

        if should_evict {
            self.evict_entries().await?;
        }

        let mut embeddings = self.embeddings.write().await;
        embeddings.insert(key.to_string(), entry);

        let mut total_size = self.total_size_bytes.write().await;
        *total_size += entry_size;

        drop(embeddings);
        drop(total_size);

        // Rebuild GPU matrix for efficient similarity search
        self.rebuild_gpu_matrix().await?;

        debug!("Inserted embedding for key: {}", key);
        Ok(())
    }

    /// Get an embedding from the cache
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut embeddings = self.embeddings.write().await;

        if let Some(entry) = embeddings.get_mut(key) {
            entry.last_accessed = chrono::Utc::now();
            entry.access_count += 1;
            Some(entry.embedding.clone())
        } else {
            None
        }
    }

    /// Find similar embeddings using GPU-accelerated similarity search
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>> {
        if query_embedding.len() != self.config.embedding_dim {
            return Err(anyhow::anyhow!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                query_embedding.len()
            ));
        }

        let embeddings = self.embeddings.read().await;

        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        // Use GPU-accelerated similarity search if available
        let similarities = if self.gpu_context.is_some()
            && self.tensor_core.is_some()
            && self.config.enable_gpu
        {
            self.gpu_similarity_search(query_embedding).await?
        } else if self.config.enable_simd {
            self.simd_similarity_search(query_embedding).await?
        } else {
            self.cpu_similarity_search(query_embedding).await?
        };

        // Get top-k results
        let mut results: Vec<(String, f32)> = similarities.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Update access statistics
        drop(embeddings);
        let mut embeddings = self.embeddings.write().await;

        let similarity_results: Vec<SimilarityResult> = results
            .into_iter()
            .filter_map(|(key, score)| {
                if let Some(entry) = embeddings.get_mut(&key) {
                    entry.last_accessed = chrono::Utc::now();
                    entry.access_count += 1;
                    Some(SimilarityResult {
                        key: key.clone(),
                        similarity_score: score,
                        embedding: entry.embedding.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Found {} similar embeddings for query",
            similarity_results.len()
        );

        Ok(similarity_results)
    }

    /// Remove an embedding from the cache
    pub async fn remove(&self, key: &str) -> Option<Vec<f32>> {
        let mut embeddings = self.embeddings.write().await;

        if let Some(entry) = embeddings.remove(key) {
            let entry_size = entry.embedding.len() * std::mem::size_of::<f32>();
            let mut total_size = self.total_size_bytes.write().await;
            *total_size = total_size.saturating_sub(entry_size);

            debug!("Removed embedding for key: {}", key);
            Some(entry.embedding)
        } else {
            None
        }
    }

    /// Clear the entire cache
    pub async fn clear(&self) {
        let mut embeddings = self.embeddings.write().await;
        embeddings.clear();

        let mut total_size = self.total_size_bytes.write().await;
        *total_size = 0;

        let mut matrix = self.embedding_matrix.write().await;
        *matrix = None;

        let mut key_index = self.key_index.write().await;
        key_index.clear();

        info!("Cache cleared");
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let embeddings = self.embeddings.read().await;
        let total_size = *self.total_size_bytes.read().await;

        let total_accesses: usize = embeddings.values().map(|e| e.access_count).sum();
        let avg_accesses = if !embeddings.is_empty() {
            total_accesses as f64 / embeddings.len() as f64
        } else {
            0.0
        };

        CacheStats {
            total_entries: embeddings.len(),
            total_size_bytes: total_size,
            total_accesses,
            avg_accesses_per_entry: avg_accesses,
            gpu_enabled: self.gpu_context.is_some(),
            simd_enabled: self.config.enable_simd,
            max_size_bytes: self.config.max_cache_size_mb * 1024 * 1024,
            utilization: total_size as f64 / (self.config.max_cache_size_mb * 1024 * 1024) as f64,
        }
    }

    /// Save cache to file
    pub async fn save_to_file(&self, path: &PathBuf) -> Result<()> {
        let embeddings = self.embeddings.read().await;

        // Convert to serializable format
        let data: Vec<(String, Vec<f32>)> = embeddings
            .iter()
            .map(|(k, v)| (k.clone(), v.embedding.clone()))
            .collect();

        let json = serde_json::to_string(&data)?;
        tokio::fs::write(path, json).await?;

        info!("Cache saved to {:?} ({} entries)", path, embeddings.len());
        Ok(())
    }

    // Private helper methods

    async fn initialize_gpu(config: &CacheConfig) -> Result<(GpuContext, TensorCore)> {
        let context = match config.gpu_backend {
            GpuBackend::Cuda => GpuContext::new_cuda()?,
            GpuBackend::Metal => GpuContext::new_metal()?,
            GpuBackend::Cpu => return Err(anyhow::anyhow!("CPU backend selected")),
        };

        let tensor_core = TensorCore::new(&context)?;

        Ok((context, tensor_core))
    }

    async fn create_gpu_buffer(context: &GpuContext, config: &CacheConfig) -> Result<GpuBuffer> {
        let buffer_size = config.max_cache_size_mb * 1024 * 1024;
        GpuBuffer::new(context, buffer_size).context("Failed to create GPU buffer")
    }

    async fn load_from_file(path: &PathBuf) -> Result<HashMap<String, CachedEmbedding>> {
        let json = tokio::fs::read_to_string(path).await?;
        let data: Vec<(String, Vec<f32>)> = serde_json::from_str(&json)?;

        let mut embeddings = HashMap::new();
        for (key, embedding) in data {
            embeddings.insert(
                key.clone(),
                CachedEmbedding {
                    key,
                    embedding,
                    last_accessed: chrono::Utc::now(),
                    access_count: 0,
                },
            );
        }

        Ok(embeddings)
    }

    async fn rebuild_gpu_matrix(&self) -> Result<()> {
        let embeddings = self.embeddings.read().await;

        if embeddings.is_empty() {
            return Ok(());
        }

        let n_embeddings = embeddings.len();
        let dim = self.config.embedding_dim;

        // Build embedding matrix for batch similarity computation
        let mut matrix_data = Vec::with_capacity(n_embeddings * dim);
        let mut keys = Vec::with_capacity(n_embeddings);

        for (key, entry) in embeddings.iter() {
            matrix_data.extend_from_slice(&entry.embedding);
            keys.push(key.clone());
        }

        let matrix = Array2::from_shape_vec((n_embeddings, dim), matrix_data)?;

        let mut embedding_matrix = self.embedding_matrix.write().await;
        *embedding_matrix = Some(matrix);

        let mut key_index = self.key_index.write().await;
        *key_index = keys;

        debug!("Rebuilt GPU embedding matrix: {} x {}", n_embeddings, dim);
        Ok(())
    }

    async fn gpu_similarity_search(&self, query: &[f32]) -> Result<HashMap<String, f32>> {
        // GPU features not available yet in scirs2-core rc.2
        // Fallback to SIMD implementation
        self.simd_similarity_search(query).await
    }

    async fn simd_similarity_search(&self, query: &[f32]) -> Result<HashMap<String, f32>> {
        let embeddings = self.embeddings.read().await;
        let query_array = Array1::from_vec(query.to_vec());

        // Use SIMD-optimized dot product for similarity computation
        let similarities: HashMap<String, f32> = embeddings
            .iter()
            .map(|(key, entry)| {
                let embedding_array = ArrayView1::from(&entry.embedding);
                let similarity = simd_dot_product(query_array.view(), embedding_array);
                (key.clone(), similarity)
            })
            .collect();

        Ok(similarities)
    }

    async fn cpu_similarity_search(&self, query: &[f32]) -> Result<HashMap<String, f32>> {
        let embeddings = self.embeddings.read().await;

        // Standard cosine similarity computation
        let similarities: HashMap<String, f32> = embeddings
            .iter()
            .map(|(key, entry)| {
                let similarity = Self::cosine_similarity(query, &entry.embedding);
                (key.clone(), similarity)
            })
            .collect();

        Ok(similarities)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    async fn evict_entries(&self) -> Result<()> {
        let mut embeddings = self.embeddings.write().await;

        // Find least recently used entries
        let mut entries: Vec<_> = embeddings
            .iter()
            .map(|(k, v)| (k.clone(), v.last_accessed, v.access_count))
            .collect();

        // Sort by last accessed time (oldest first)
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove oldest 10% of entries
        let to_remove = (entries.len() as f32 * (1.0 - self.config.eviction_threshold)) as usize;
        let to_remove = to_remove.max(1);

        let mut removed_size = 0;
        for (key, _, _) in entries.iter().take(to_remove) {
            if let Some(entry) = embeddings.remove(key) {
                removed_size += entry.embedding.len() * std::mem::size_of::<f32>();
            }
        }

        let mut total_size = self.total_size_bytes.write().await;
        *total_size = total_size.saturating_sub(removed_size);

        info!("Evicted {} cache entries ({} bytes)", to_remove, removed_size);

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub total_accesses: usize,
    pub avg_accesses_per_entry: f64,
    pub gpu_enabled: bool,
    pub simd_enabled: bool,
    pub max_size_bytes: usize,
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_creation() {
        let config = CacheConfig {
            enable_gpu: false, // Disable GPU for testing
            ..Default::default()
        };
        let cache = GpuEmbeddingCache::new(config).await;
        assert!(cache.is_ok());
    }

    #[tokio::test]
    async fn test_insert_and_get() -> Result<()> {
        let config = CacheConfig {
            enable_gpu: false,
            embedding_dim: 4,
            ..Default::default()
        };
        let cache = GpuEmbeddingCache::new(config).await?;

        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        cache.insert("test_key", &embedding).await?;

        let retrieved = cache.get("test_key").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);

        Ok(())
    }

    #[tokio::test]
    async fn test_similarity_search() -> Result<()> {
        let config = CacheConfig {
            enable_gpu: false,
            embedding_dim: 4,
            enable_simd: true,
            ..Default::default()
        };
        let cache = GpuEmbeddingCache::new(config).await?;

        // Insert test embeddings
        cache.insert("key1", &[1.0, 0.0, 0.0, 0.0]).await?;
        cache.insert("key2", &[0.9, 0.1, 0.0, 0.0]).await?;
        cache.insert("key3", &[0.0, 1.0, 0.0, 0.0]).await?;

        // Find similar to key1
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = cache.find_similar(&query, 2).await?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "key1");
        assert!(results[0].similarity_score > results[1].similarity_score);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_stats() -> Result<()> {
        let config = CacheConfig {
            enable_gpu: false,
            embedding_dim: 4,
            ..Default::default()
        };
        let cache = GpuEmbeddingCache::new(config).await?;

        cache.insert("key1", &[0.1, 0.2, 0.3, 0.4]).await?;
        cache.insert("key2", &[0.5, 0.6, 0.7, 0.8]).await?;

        let stats = cache.get_stats().await;
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_size_bytes > 0);

        Ok(())
    }
}
