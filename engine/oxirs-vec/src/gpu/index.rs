//! GPU-accelerated vector index implementations

use super::{GpuAccelerator, GpuConfig, GpuMemoryPool, GpuPerformanceStats};
use crate::{similarity::SimilarityMetric, Vector, VectorData, VectorPrecision};
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// GPU-accelerated vector index
#[derive(Debug)]
pub struct GpuVectorIndex {
    accelerator: Arc<GpuAccelerator>,
    vectors: Vec<Vector>,
    vector_data: Vec<f32>,
    dimension: usize,
    memory_pool: Arc<RwLock<GpuMemoryPool>>,
    uri_map: HashMap<String, usize>,
}

impl GpuVectorIndex {
    /// Create a new GPU vector index
    pub fn new(config: GpuConfig) -> Result<Self> {
        let accelerator = Arc::new(GpuAccelerator::new(config.clone())?);
        let memory_pool = Arc::new(RwLock::new(GpuMemoryPool::new(&config, 1024)?));

        Ok(Self {
            accelerator,
            vectors: Vec::new(),
            vector_data: Vec::new(),
            dimension: 0,
            memory_pool,
            uri_map: HashMap::new(),
        })
    }

    /// Add vectors to the index
    pub fn add_vectors(&mut self, vectors: Vec<Vector>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Set dimension from first vector if not set
        if self.dimension == 0 {
            self.dimension = vectors[0].dimensions;
        }

        // Validate all vectors have the same dimension
        for vector in &vectors {
            if vector.dimensions != self.dimension {
                return Err(anyhow!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    vector.dimensions
                ));
            }
        }

        // Flatten vector data for GPU processing
        for vector in &vectors {
            match &vector.values {
                VectorData::F32(data) => self.vector_data.extend(data),
                VectorData::F64(data) => {
                    // Convert f64 to f32 for GPU processing
                    self.vector_data.extend(data.iter().map(|&x| x as f32));
                }
                _ => return Err(anyhow!("Unsupported vector precision for GPU processing")),
            }
        }

        self.vectors.extend(vectors);
        Ok(())
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query: &Vector,
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let query_data = match &query.values {
            VectorData::F32(data) => data.clone(),
            VectorData::F64(data) => data.iter().map(|&x| x as f32).collect(),
            _ => {
                return Err(anyhow!(
                    "Unsupported query vector precision for GPU processing"
                ))
            }
        };

        if query.dimensions != self.dimension {
            return Err(anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.dimensions
            ));
        }

        // Compute similarities using GPU
        let similarities = self.accelerator.compute_similarity(
            &query_data,
            &self.vector_data,
            1,
            self.vectors.len(),
            self.dimension,
            metric,
        )?;

        // Sort and return top-k results
        let mut results: Vec<(usize, f32)> = similarities
            .into_iter()
            .enumerate()
            .map(|(idx, sim)| (idx, sim))
            .collect();

        // Sort by similarity (descending for similarity, ascending for distance)
        match metric {
            SimilarityMetric::Cosine | SimilarityMetric::Pearson | SimilarityMetric::Jaccard => {
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        results.truncate(k);
        Ok(results)
    }

    /// Batch search for multiple queries
    pub fn batch_search(
        &self,
        queries: &[Vector],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        let mut results = Vec::new();

        for query in queries {
            let query_results = self.search(query, k, metric)?;
            results.push(query_results);
        }

        Ok(results)
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the dimension of vectors in this index
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> Arc<parking_lot::RwLock<GpuPerformanceStats>> {
        self.accelerator.performance_stats()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.vector_data.clear();
        self.dimension = 0;
        self.accelerator.reset_stats();
    }
}

impl crate::VectorIndex for GpuVectorIndex {
    fn insert(&mut self, uri: String, vector: crate::Vector) -> Result<()> {
        // Store the URI mapping
        let index = self.vectors.len();
        self.uri_map.insert(uri, index);
        self.add_vectors(vec![vector])?;
        Ok(())
    }

    fn search_knn(&self, query: &crate::Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let results = self.search(query, k, SimilarityMetric::Cosine)?;
        Ok(results
            .into_iter()
            .filter_map(|(index, score)| {
                // Find URI by index (reverse lookup)
                self.uri_map
                    .iter()
                    .find(|(_, &idx)| idx == index)
                    .map(|(uri, _)| (uri.clone(), score))
            })
            .collect())
    }

    fn search_threshold(
        &self,
        query: &crate::Vector,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        // Use large k to get many candidates, then filter by threshold
        let results = self.search(query, 1000, SimilarityMetric::Cosine)?;
        Ok(results
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .filter_map(|(index, score)| {
                self.uri_map
                    .iter()
                    .find(|(_, &idx)| idx == index)
                    .map(|(uri, _)| (uri.clone(), score))
            })
            .collect())
    }

    fn get_vector(&self, uri: &str) -> Option<&crate::Vector> {
        if let Some(&index) = self.uri_map.get(uri) {
            self.vectors.get(index)
        } else {
            None
        }
    }
}

/// Advanced GPU vector index with additional optimizations
#[derive(Debug)]
pub struct AdvancedGpuVectorIndex {
    base_index: GpuVectorIndex,
    enable_quantization: bool,
    quantization_bits: u8,
    use_tensor_cores: bool,
}

impl AdvancedGpuVectorIndex {
    /// Create a new advanced GPU vector index
    pub fn new(mut config: GpuConfig) -> Result<Self> {
        config.enable_tensor_cores = true;
        config.enable_mixed_precision = true;

        let base_index = GpuVectorIndex::new(config)?;

        Ok(Self {
            base_index,
            enable_quantization: false,
            quantization_bits: 8,
            use_tensor_cores: true,
        })
    }

    /// Enable quantization for memory efficiency
    pub fn enable_quantization(&mut self, bits: u8) {
        self.enable_quantization = true;
        self.quantization_bits = bits;
    }

    /// Optimized batch processing for large-scale operations
    pub fn batch_process(
        &self,
        queries: &[Vector],
        batch_size: usize,
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        let mut all_results = Vec::new();

        for batch in queries.chunks(batch_size) {
            let batch_results = self.base_index.batch_search(batch, k, metric)?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> Result<MemoryUsageStats> {
        let device = self.base_index.accelerator.device();
        let pool_stats = self.base_index.memory_pool.read().stats();

        Ok(MemoryUsageStats {
            total_gpu_memory: device.total_memory,
            free_gpu_memory: device.free_memory,
            used_by_index: pool_stats.used_memory,
            vector_count: self.base_index.len(),
            dimension: self.base_index.dimension(),
            memory_per_vector: if self.base_index.len() > 0 {
                pool_stats.used_memory / self.base_index.len()
            } else {
                0
            },
        })
    }
}

/// Memory usage statistics for GPU vector index
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_gpu_memory: usize,
    pub free_gpu_memory: usize,
    pub used_by_index: usize,
    pub vector_count: usize,
    pub dimension: usize,
    pub memory_per_vector: usize,
}

impl MemoryUsageStats {
    /// Get memory utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total_gpu_memory > 0 {
            (self.total_gpu_memory - self.free_gpu_memory) as f64 / self.total_gpu_memory as f64
        } else {
            0.0
        }
    }

    /// Print memory statistics
    pub fn print(&self) {
        println!("GPU Vector Index Memory Usage:");
        println!(
            "  Total GPU Memory: {:.2} GB",
            self.total_gpu_memory as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!(
            "  Free GPU Memory: {:.2} GB",
            self.free_gpu_memory as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!(
            "  Used by Index: {:.2} MB",
            self.used_by_index as f64 / 1024.0 / 1024.0
        );
        println!("  Vectors: {} ({}D)", self.vector_count, self.dimension);
        println!(
            "  Memory per Vector: {:.2} KB",
            self.memory_per_vector as f64 / 1024.0
        );
        println!("  GPU Utilization: {:.1}%", self.utilization() * 100.0);
    }
}

/// Batch vector processor for high-throughput operations
#[derive(Debug)]
pub struct BatchVectorProcessor {
    accelerator: Arc<GpuAccelerator>,
    batch_size: usize,
    max_concurrent_batches: usize,
}

impl BatchVectorProcessor {
    /// Create a new batch vector processor
    pub fn new(config: GpuConfig, batch_size: usize) -> Result<Self> {
        let accelerator = Arc::new(GpuAccelerator::new(config)?);
        let max_concurrent_batches = 4; // Adjust based on GPU memory

        Ok(Self {
            accelerator,
            batch_size,
            max_concurrent_batches,
        })
    }

    /// Process vectors in batches with specified operation
    pub fn process_batches<F, R>(&self, vectors: &[Vector], operation: F) -> Result<Vec<R>>
    where
        F: Fn(&[Vector]) -> Result<Vec<R>> + Send + Sync,
        R: Send,
    {
        let mut results = Vec::new();

        for batch in vectors.chunks(self.batch_size) {
            let batch_results = operation(batch)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Parallel batch processing using multiple streams
    pub fn parallel_process_batches<F, R>(&self, vectors: &[Vector], operation: F) -> Result<Vec<R>>
    where
        F: Fn(&[Vector]) -> Result<Vec<R>> + Send + Sync + Clone + 'static,
        R: Send + 'static,
    {
        use std::thread;

        let chunks: Vec<&[Vector]> = vectors.chunks(self.batch_size).collect();
        let mut handles = Vec::new();
        let mut results = Vec::new();

        for chunk_batch in chunks.chunks(self.max_concurrent_batches) {
            for chunk in chunk_batch {
                let chunk_vec = chunk.to_vec();
                let op = operation.clone();

                let handle = thread::spawn(move || op(&chunk_vec));
                handles.push(handle);
            }

            // Collect results from this batch of threads
            for handle in handles.drain(..) {
                match handle.join() {
                    Ok(Ok(batch_results)) => results.extend(batch_results),
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err(anyhow!("Thread panicked during batch processing")),
                }
            }
        }

        Ok(results)
    }

    /// Get processing throughput (vectors per second)
    pub fn throughput(&self, vectors_processed: usize, duration: std::time::Duration) -> f64 {
        if duration.as_secs_f64() > 0.0 {
            vectors_processed as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }
}
