//! Memory optimization for AI operations
//!
//! This module provides memory-efficient operations for embeddings, model weights,
//! and large-scale AI processing to minimize memory footprint in production.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

pub mod pooling;
pub mod streaming;
pub mod compression;
pub mod tensor_ops;

pub use pooling::{MemoryPool, PooledBuffer};
pub use streaming::{StreamProcessor, ChunkProcessor};
pub use compression::{Compressor, CompressionAlgorithm};
pub use tensor_ops::{TensorOptimizer, MemoryEfficientTensor};

/// Configuration for memory optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,

    /// Pool size in MB
    pub pool_size_mb: usize,

    /// Enable streaming for large datasets
    pub enable_streaming: bool,

    /// Chunk size for streaming (in records)
    pub streaming_chunk_size: usize,

    /// Enable compression for cached data
    pub enable_compression: bool,

    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,

    /// Enable tensor optimization
    pub enable_tensor_optimization: bool,

    /// Use low-precision for inference (f16 instead of f32)
    pub use_low_precision: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_size_mb: 512, // 512MB default pool
            enable_streaming: true,
            streaming_chunk_size: 1000,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            enable_tensor_optimization: true,
            use_low_precision: false, // f32 by default for accuracy
        }
    }
}

/// Memory optimization manager
pub struct MemoryOptimizer {
    config: MemoryOptimizationConfig,
    pool: Option<Arc<RwLock<MemoryPool>>>,
    compressor: Option<Compressor>,
    tensor_optimizer: Option<TensorOptimizer>,
    metrics: Arc<RwLock<MemoryMetrics>>,
}

/// Memory usage metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub compression_ratio: f64,
    pub pool_hits: u64,
    pub pool_misses: u64,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        let pool = if config.enable_pooling {
            Some(Arc::new(RwLock::new(MemoryPool::new(
                config.pool_size_mb * 1024 * 1024,
            ))))
        } else {
            None
        };

        let compressor = if config.enable_compression {
            Some(Compressor::new(config.compression_algorithm))
        } else {
            None
        };

        let tensor_optimizer = if config.enable_tensor_optimization {
            Some(TensorOptimizer::new(config.use_low_precision))
        } else {
            None
        };

        Ok(Self {
            config,
            pool,
            compressor,
            tensor_optimizer,
            metrics: Arc::new(RwLock::new(MemoryMetrics::default())),
        })
    }

    /// Allocate memory from pool if available, otherwise use heap
    pub fn allocate(&self, size: usize) -> Result<PooledBuffer> {
        if let Some(ref pool) = self.pool {
            let mut pool_guard = pool
                .write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

            match pool_guard.allocate(size) {
                Ok(buffer) => {
                    // Update metrics
                    let mut metrics = self
                        .metrics
                        .write()
                        .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
                    metrics.pool_hits += 1;
                    metrics.total_allocated += size;
                    metrics.current_usage += size;
                    if metrics.current_usage > metrics.peak_usage {
                        metrics.peak_usage = metrics.current_usage;
                    }

                    Ok(buffer)
                }
                Err(_) => {
                    // Pool exhausted, allocate on heap
                    let mut metrics = self
                        .metrics
                        .write()
                        .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
                    metrics.pool_misses += 1;

                    PooledBuffer::new_heap(size)
                }
            }
        } else {
            // Pooling disabled, allocate on heap
            PooledBuffer::new_heap(size)
        }
    }

    /// Compress data if compression is enabled
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if let Some(ref compressor) = self.compressor {
            let compressed = compressor.compress(data)?;

            // Update metrics
            let mut metrics = self
                .metrics
                .write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
            let ratio = data.len() as f64 / compressed.len() as f64;
            metrics.compression_ratio = ratio;

            Ok(compressed)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data if compression is enabled
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if let Some(ref compressor) = self.compressor {
            compressor.decompress(data)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Optimize tensor for memory efficiency
    pub fn optimize_tensor(&self, tensor: &[f32]) -> Result<MemoryEfficientTensor> {
        if let Some(ref optimizer) = self.tensor_optimizer {
            optimizer.optimize(tensor)
        } else {
            // No optimization, wrap as-is
            Ok(MemoryEfficientTensor::F32(tensor.to_vec()))
        }
    }

    /// Get current memory metrics
    pub fn metrics(&self) -> Result<MemoryMetrics> {
        let metrics = self
            .metrics
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(metrics.clone())
    }

    /// Reset metrics
    pub fn reset_metrics(&self) -> Result<()> {
        let mut metrics = self
            .metrics
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *metrics = MemoryMetrics::default();
        Ok(())
    }

    /// Get pool statistics
    pub fn pool_hit_rate(&self) -> Result<f64> {
        let metrics = self
            .metrics
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let total = metrics.pool_hits + metrics.pool_misses;
        if total == 0 {
            return Ok(0.0);
        }

        Ok(metrics.pool_hits as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimizer_creation() {
        let optimizer = MemoryOptimizer::new(MemoryOptimizationConfig::default()).unwrap();
        let metrics = optimizer.metrics().unwrap();
        assert_eq!(metrics.total_allocated, 0);
    }

    #[test]
    fn test_memory_allocation() {
        let optimizer = MemoryOptimizer::new(MemoryOptimizationConfig::default()).unwrap();

        let buffer = optimizer.allocate(1024).unwrap();
        assert!(buffer.len() >= 1024);

        let metrics = optimizer.metrics().unwrap();
        assert_eq!(metrics.pool_hits, 1);
        assert_eq!(metrics.total_allocated, 1024);
    }

    #[test]
    fn test_compression() {
        let optimizer = MemoryOptimizer::new(MemoryOptimizationConfig::default()).unwrap();

        let data = vec![42u8; 1000];
        let compressed = optimizer.compress(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = optimizer.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_pool_hit_rate() {
        let optimizer = MemoryOptimizer::new(MemoryOptimizationConfig::default()).unwrap();

        // Allocate from pool
        let _b1 = optimizer.allocate(1024).unwrap();
        let _b2 = optimizer.allocate(2048).unwrap();

        let hit_rate = optimizer.pool_hit_rate().unwrap();
        assert!(hit_rate > 0.0);
    }
}
