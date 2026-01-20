//! GPU Acceleration for Embedding Computations
//!
//! This module provides GPU-accelerated implementations of embedding operations
//! using scirs2-linalg GPU features for CUDA, OpenCL, ROCm, and Metal backends.

use crate::models::common::*;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
#[cfg(feature = "gpu")]
use std::collections::VecDeque;
#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex, RwLock};
#[cfg(feature = "gpu")]
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
// TODO: scirs2_linalg::gpu module is not yet available
// Enable this when the GPU module is implemented in scirs2_linalg
// use scirs2_linalg::gpu::{GpuArray, GpuContext, GpuError};
// Placeholder types until scirs2_linalg::gpu is available
pub type GpuArray<T> = Vec<T>;
#[cfg(feature = "gpu")]
pub type GpuContext = ();
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuError(String);

#[cfg(feature = "gpu")]
impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for GpuError {}

/// Memory pool for GPU buffers
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuMemoryPool {
    available_buffers: VecDeque<GpuArray<f32>>,
    buffer_size: usize,
    total_allocated: AtomicU64,
    peak_usage: AtomicU64,
}

/// Adaptive batch sizing configuration
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct AdaptiveBatchConfig {
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub target_gpu_utilization: f32,
    pub memory_usage_threshold: f32,
}

/// Enhanced GPU-accelerated embedding computations with memory pooling and adaptive batching
#[cfg(feature = "gpu")]
pub struct GpuEmbeddingAccelerator {
    context: GpuContext,
    device_id: u32,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
    batch_config: AdaptiveBatchConfig,
    performance_stats: Arc<RwLock<GpuPerformanceStats>>,
    optimal_batch_size: Arc<AtomicU64>,
}

/// GPU performance statistics
#[cfg(feature = "gpu")]
#[derive(Debug, Default)]
pub struct GpuPerformanceStats {
    pub total_operations: u64,
    pub total_compute_time: Duration,
    pub memory_transfers: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_batch_size: f32,
    pub gpu_utilization_percentage: f32,
}

/// Comprehensive GPU performance report
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuPerformanceReport {
    pub device_id: u32,
    pub total_operations: u64,
    pub average_compute_time: Duration,
    pub gpu_utilization: f32,
    pub memory_allocated_mb: f64,
    pub memory_peak_mb: f64,
    pub cache_hit_rate: f32,
    pub optimal_batch_size: usize,
}

#[cfg(feature = "gpu")]
impl GpuMemoryPool {
    pub fn new(buffer_size: usize, initial_pool_size: usize) -> Self {
        Self {
            available_buffers: VecDeque::with_capacity(initial_pool_size),
            buffer_size,
            total_allocated: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
        }
    }

    pub fn get_buffer(&mut self) -> Option<GpuArray<f32>> {
        self.available_buffers.pop_front()
    }

    pub fn return_buffer(&mut self, buffer: GpuArray<f32>) {
        if buffer.len() == self.buffer_size {
            self.available_buffers.push_back(buffer);
        }
        // If buffer size doesn't match, let it drop (auto-deallocate)
    }

    pub fn get_memory_stats(&self) -> (u64, u64) {
        (
            self.total_allocated.load(Ordering::Relaxed),
            self.peak_usage.load(Ordering::Relaxed),
        )
    }
}

#[cfg(feature = "gpu")]
impl GpuEmbeddingAccelerator {
    /// Create a new enhanced GPU accelerator with memory pooling and adaptive batching
    /// Note: Currently using placeholder until scirs2_linalg::gpu is available
    pub fn new(device_id: u32) -> Result<Self, GpuError> {
        let context = (); // Placeholder GpuContext

        let memory_pool = Arc::new(Mutex::new(GpuMemoryPool::new(1024 * 1024, 10))); // 1MB buffers, 10 initial

        let batch_config = AdaptiveBatchConfig {
            min_batch_size: 32,
            max_batch_size: 8192,
            target_gpu_utilization: 0.85,
            memory_usage_threshold: 0.8,
        };

        Ok(Self {
            context,
            device_id,
            memory_pool,
            batch_config,
            performance_stats: Arc::new(RwLock::new(GpuPerformanceStats::default())),
            optimal_batch_size: Arc::new(AtomicU64::new(512)), // Start with reasonable default
        })
    }

    /// Get optimal batch size based on recent performance
    pub async fn get_optimal_batch_size(&self, data_size: usize) -> usize {
        let optimal = self.optimal_batch_size.load(Ordering::Relaxed) as usize;
        let config_min = self.batch_config.min_batch_size;
        let config_max = self.batch_config.max_batch_size;

        // Clamp to configuration bounds and data size
        optimal.clamp(config_min, config_max.min(data_size))
    }

    /// Update optimal batch size based on performance feedback
    pub async fn update_batch_size_feedback(&self, _batch_size: usize, performance_score: f32) {
        let current_optimal = self.optimal_batch_size.load(Ordering::Relaxed) as usize;

        // Simple adaptive algorithm: increase if performance is good, decrease if poor
        let new_optimal = if performance_score > 0.8 {
            // Good performance, try larger batches
            (current_optimal as f32 * 1.1).round() as usize
        } else if performance_score < 0.5 {
            // Poor performance, try smaller batches
            (current_optimal as f32 * 0.9).round() as usize
        } else {
            current_optimal
        };

        let clamped_optimal = new_optimal.clamp(
            self.batch_config.min_batch_size,
            self.batch_config.max_batch_size,
        );

        self.optimal_batch_size
            .store(clamped_optimal as u64, Ordering::Relaxed);
    }

    /// GPU-accelerated batch distance computation
    pub fn batch_l2_distances_gpu(
        &self,
        vectors_a: &[Array1<f64>],
        vectors_b: &[Array1<f64>],
    ) -> Result<Vec<f64>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, use CPU implementation as fallback
        let mut distances = Vec::with_capacity(vectors_a.len());
        for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
            let dist: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            distances.push(dist);
        }
        Ok(distances)
    }

    /// GPU-accelerated cosine similarity matrix
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    pub fn cosine_similarity_matrix_gpu(
        &self,
        vectors: &[Array1<f64>],
    ) -> Result<Array2<f64>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, use CPU implementation as fallback
        use scirs2_core::ndarray_ext::Array2;

        let n = vectors.len();
        let mut similarity_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let dot: f64 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let norm_i: f64 = vectors[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_j: f64 = vectors[j].iter().map(|x| x * x).sum::<f64>().sqrt();
                similarity_matrix[[i, j]] = dot / (norm_i * norm_j + 1e-8);
            }
        }
        Ok(similarity_matrix)
    }

    /// GPU-accelerated gradient updates for large embedding matrices
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    pub fn batch_gradient_update_gpu(
        &self,
        embeddings: &mut [Array2<f64>],
        gradients: &[Array2<f64>],
        learning_rate: f64,
        l2_reg: f64,
    ) -> Result<(), GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, use CPU implementation as fallback
        for (embedding, gradient) in embeddings.iter_mut().zip(gradients.iter()) {
            // Apply gradient update with L2 regularization
            for (emb, grad) in embedding.iter_mut().zip(gradient.iter()) {
                *emb -= learning_rate * (grad + l2_reg * *emb);
            }
        }
        Ok(())
    }

    /// Advanced GPU-accelerated adaptive batch processing with memory pooling
    pub async fn adaptive_batch_processing<T, R>(
        &self,
        data: &[T],
        mut process_fn: impl FnMut(&[T]) -> Result<Vec<R>, GpuError>,
    ) -> Result<Vec<R>, GpuError> {
        let start_time = Instant::now();
        let batch_size = self.get_optimal_batch_size(data.len()).await;

        let mut results = Vec::with_capacity(data.len());
        let mut total_processing_time = Duration::ZERO;

        for chunk in data.chunks(batch_size) {
            let chunk_start = Instant::now();
            let chunk_results = process_fn(chunk)?;
            let chunk_time = chunk_start.elapsed();

            results.extend(chunk_results);
            total_processing_time += chunk_time;
        }

        // Calculate performance score and update batch size
        let total_time = start_time.elapsed();
        let gpu_utilization = total_processing_time.as_secs_f32() / total_time.as_secs_f32();
        let performance_score = gpu_utilization.min(1.0);

        self.update_batch_size_feedback(batch_size, performance_score)
            .await;

        // Update performance statistics
        let mut stats = self.performance_stats.write().unwrap();
        stats.total_operations += 1;
        stats.total_compute_time += total_time;
        stats.gpu_utilization_percentage = gpu_utilization * 100.0;
        stats.average_batch_size = (stats.average_batch_size + batch_size as f32) / 2.0;

        Ok(results)
    }

    /// GPU-accelerated matrix multiplication with memory reuse
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    pub async fn optimized_matrix_multiply(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> Result<Array2<f32>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, use CPU implementation as fallback
        let result = a.dot(b);

        Ok(result)
    }

    /// High-performance embedding search with GPU acceleration
    pub async fn gpu_embedding_search(
        &self,
        query_embedding: &Array1<f32>,
        database_embeddings: &[Array1<f32>],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, GpuError> {
        // Use adaptive batching for large databases
        let batch_size = self.get_optimal_batch_size(database_embeddings.len()).await;
        let mut all_similarities = Vec::with_capacity(database_embeddings.len());

        // Process in adaptive batches
        for (batch_idx, batch) in database_embeddings.chunks(batch_size).enumerate() {
            let similarities = self
                .compute_batch_similarities(query_embedding, batch)
                .await?;

            for (local_idx, similarity) in similarities.iter().enumerate() {
                let global_idx = batch_idx * batch_size + local_idx;
                all_similarities.push((global_idx, *similarity));
            }
        }

        // Sort and return top-k
        all_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_similarities.truncate(top_k);

        Ok(all_similarities)
    }

    /// Compute similarities for a batch with GPU acceleration
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    async fn compute_batch_similarities(
        &self,
        query: &Array1<f32>,
        batch: &[Array1<f32>],
    ) -> Result<Vec<f32>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, use CPU implementation as fallback
        let mut similarities = Vec::with_capacity(batch.len());

        for emb in batch {
            // Compute cosine similarity: (a · b) / (||a|| * ||b||)
            let dot_product: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let norm_query: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_emb: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let similarity = dot_product / (norm_query * norm_emb + 1e-8);
            similarities.push(similarity);
        }

        Ok(similarities)
    }

    /// GPU-accelerated Xavier initialization for large embedding matrices
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    pub fn xavier_init_gpu(
        &self,
        shapes: &[(usize, usize)],
        fan_in: usize,
        fan_out: usize,
        seed: u64,
    ) -> Result<Vec<Array2<f64>>, GpuError> {
        use scirs2_core::random::Random;

        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let mut rng = Random::seed(seed);
        let scale = 2.0 * limit;

        let mut results = Vec::with_capacity(shapes.len());
        for &(rows, cols) in shapes {
            // Generate uniform random numbers in [-limit, limit]
            let data: Vec<f64> = (0..rows * cols)
                .map(|_| rng.random_f64() * scale - limit)
                .collect();
            let array = Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| GpuError(format!("Failed to create array: {}", e)))?;
            results.push(array);
        }
        Ok(results)
    }

    /// GPU-accelerated contrastive learning updates
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    pub fn contrastive_learning_gpu(
        &self,
        _entity_embeddings: &mut [Array1<f32>],
        _similarity_pairs: &[(usize, usize)],
        _negative_samples: &[(usize, usize)],
        _temperature: f32,
        _learning_rate: f32,
    ) -> Result<f32, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        // For now, return placeholder loss
        Ok(0.0)
    }

    /// Helper function to upload vectors to GPU
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    fn upload_vectors_to_gpu(&self, _vectors: &[Array1<f64>]) -> Result<GpuArray<f64>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        Ok(Vec::new())
    }

    /// Helper function to upload f32 vectors to GPU
    /// Note: Currently using CPU fallback until scirs2_linalg::gpu is available
    fn upload_f32_vectors_to_gpu(
        &self,
        _vectors: &[Array1<f32>],
    ) -> Result<GpuArray<f32>, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        Ok(Vec::new())
    }

    /// Get GPU device info
    pub fn device_info(&self) -> String {
        format!(
            "GPU Device {} (placeholder - scirs2_linalg::gpu not yet available)",
            self.device_id
        )
    }

    /// Get available GPU memory
    /// Note: Currently using placeholder until scirs2_linalg::gpu is available
    pub fn available_memory(&self) -> Result<u64, GpuError> {
        // TODO: Use actual GPU operations when scirs2_linalg::gpu is available
        Ok(0)
    }

    /// GPU memory and performance monitoring
    pub async fn get_performance_report(&self) -> GpuPerformanceReport {
        let stats = self.performance_stats.read().unwrap();
        let (allocated, peak) = {
            let pool = self.memory_pool.lock().unwrap();
            pool.get_memory_stats()
        };

        GpuPerformanceReport {
            device_id: self.device_id,
            total_operations: stats.total_operations,
            average_compute_time: if stats.total_operations > 0 {
                stats.total_compute_time / stats.total_operations as u32
            } else {
                Duration::ZERO
            },
            gpu_utilization: stats.gpu_utilization_percentage,
            memory_allocated_mb: allocated as f64 / (1024.0 * 1024.0),
            memory_peak_mb: peak as f64 / (1024.0 * 1024.0),
            cache_hit_rate: if stats.cache_hits + stats.cache_misses > 0 {
                stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32
            } else {
                0.0
            },
            optimal_batch_size: self.optimal_batch_size.load(Ordering::Relaxed) as usize,
        }
    }

    /// Reset performance statistics
    pub fn reset_performance_stats(&self) {
        let mut stats = self.performance_stats.write().unwrap();
        *stats = GpuPerformanceStats::default();
        self.optimal_batch_size.store(512, Ordering::Relaxed);
    }

    /// Get current memory pool status
    pub fn get_memory_pool_status(&self) -> (usize, u64, u64) {
        let pool = self.memory_pool.lock().unwrap();
        let (allocated, peak) = pool.get_memory_stats();
        (pool.available_buffers.len(), allocated, peak)
    }
}

/// CPU fallback implementations when GPU is not available
#[cfg(not(feature = "gpu"))]
use scirs2_core::random::Random;

#[cfg(not(feature = "gpu"))]
pub struct GpuEmbeddingAccelerator;

#[cfg(not(feature = "gpu"))]
impl GpuEmbeddingAccelerator {
    pub fn new(_device_id: u32) -> Result<Self> {
        Ok(Self)
    }

    /// Fallback to CPU implementation
    pub fn batch_l2_distances_gpu(
        &self,
        vectors_a: &[Array1<f64>],
        vectors_b: &[Array1<f64>],
    ) -> Result<Vec<f64>> {
        Ok(batch_l2_distances(vectors_a, vectors_b))
    }

    /// Fallback to CPU implementation
    pub fn cosine_similarity_matrix_gpu(&self, vectors: &[Array1<f64>]) -> Result<Array2<f64>> {
        Ok(pairwise_distances(vectors))
    }

    /// Fallback to CPU implementation
    pub fn batch_gradient_update_gpu(
        &self,
        embeddings: &mut [Array2<f64>],
        gradients: &[Array2<f64>],
        learning_rate: f64,
        l2_reg: f64,
    ) -> Result<()> {
        batch_gradient_update(embeddings, gradients, learning_rate, l2_reg);
        Ok(())
    }

    /// Fallback to CPU implementation
    pub fn xavier_init_gpu(
        &self,
        shapes: &[(usize, usize)],
        fan_in: usize,
        fan_out: usize,
        _seed: u64,
    ) -> Result<Vec<Array2<f64>>> {
        let mut rng = Random::default();
        Ok(batch_xavier_init(shapes, fan_in, fan_out, &mut rng))
    }

    pub fn device_info(&self) -> String {
        "CPU (GPU acceleration not available)".to_string()
    }

    pub fn available_memory(&self) -> Result<u64> {
        // Return available system RAM as approximation
        Ok(8 * 1024 * 1024 * 1024) // 8GB default
    }
}

/// Adaptive acceleration that chooses between GPU and CPU based on problem size
pub struct AdaptiveEmbeddingAccelerator {
    gpu_accelerator: Option<GpuEmbeddingAccelerator>,
    gpu_threshold: usize,
}

impl AdaptiveEmbeddingAccelerator {
    /// Create adaptive accelerator with optional GPU support
    pub fn new(device_id: Option<u32>, gpu_threshold: usize) -> Result<Self> {
        #[allow(unused_variables)]
        let gpu_accelerator = if let Some(id) = device_id {
            #[cfg(feature = "gpu")]
            {
                GpuEmbeddingAccelerator::new(id).ok()
            }
            #[cfg(not(feature = "gpu"))]
            {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gpu_accelerator,
            gpu_threshold,
        })
    }

    /// Intelligently choose between GPU and CPU for distance computation
    pub fn adaptive_batch_distances(
        &self,
        vectors_a: &[Array1<f64>],
        vectors_b: &[Array1<f64>],
    ) -> Result<Vec<f64>> {
        if self.should_use_gpu(vectors_a.len() * vectors_b.len()) {
            if let Some(ref gpu) = self.gpu_accelerator {
                return gpu
                    .batch_l2_distances_gpu(vectors_a, vectors_b)
                    .map_err(|e| anyhow::anyhow!("GPU error: {:?}", e));
            }
        }

        // Fallback to optimized CPU implementation
        Ok(batch_l2_distances(vectors_a, vectors_b))
    }

    /// Intelligently choose between GPU and CPU for gradient updates
    pub fn adaptive_gradient_update(
        &self,
        embeddings: &mut [Array2<f64>],
        gradients: &[Array2<f64>],
        learning_rate: f64,
        l2_reg: f64,
    ) -> Result<()> {
        let total_elements: usize = embeddings.iter().map(|e| e.len()).sum();

        if self.should_use_gpu(total_elements) {
            if let Some(ref gpu) = self.gpu_accelerator {
                return gpu
                    .batch_gradient_update_gpu(embeddings, gradients, learning_rate, l2_reg)
                    .map_err(|e| anyhow::anyhow!("GPU error: {:?}", e));
            }
        }

        // Fallback to optimized CPU implementation
        batch_gradient_update(embeddings, gradients, learning_rate, l2_reg);
        Ok(())
    }

    /// Check if GPU should be used based on problem size
    fn should_use_gpu(&self, problem_size: usize) -> bool {
        self.gpu_accelerator.is_some() && problem_size >= self.gpu_threshold
    }

    /// Get acceleration info
    pub fn info(&self) -> String {
        match &self.gpu_accelerator {
            Some(gpu) => format!(
                "Adaptive: {} (threshold: {})",
                gpu.device_info(),
                self.gpu_threshold
            ),
            None => format!("Adaptive: CPU only (threshold: {})", self.gpu_threshold),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_accelerator_creation() {
        let accelerator = AdaptiveEmbeddingAccelerator::new(None, 1000).unwrap();
        assert!(accelerator.info().contains("CPU only"));
    }

    #[test]
    fn test_fallback_distance_computation() {
        let accelerator = AdaptiveEmbeddingAccelerator::new(None, 1000).unwrap();

        let vectors_a = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
        ];
        let vectors_b = vec![
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
            Array1::from_vec(vec![10.0, 11.0, 12.0]),
        ];

        let distances = accelerator
            .adaptive_batch_distances(&vectors_a, &vectors_b)
            .unwrap();
        assert_eq!(distances.len(), 4); // 2x2 combinations
    }

    #[test]
    fn test_fallback_gradient_update() {
        let accelerator = AdaptiveEmbeddingAccelerator::new(None, 1000).unwrap();

        let mut embeddings = vec![Array2::zeros((2, 3))];
        let gradients = vec![Array2::ones((2, 3))];

        accelerator
            .adaptive_gradient_update(&mut embeddings, &gradients, 0.01, 0.001)
            .unwrap();

        // Check that gradients were applied
        assert!(embeddings[0][[0, 0]] != 0.0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_accelerator_creation() {
        // This test will only run when GPU features are enabled
        match GpuEmbeddingAccelerator::new(0) {
            Ok(gpu) => {
                println!("GPU Accelerator: {}", gpu.device_info());
                let memory = gpu.available_memory().unwrap_or(0);
                println!("Available GPU Memory: {} MB", memory / (1024 * 1024));
            }
            Err(_) => {
                println!("GPU not available for testing");
            }
        }
    }
}
