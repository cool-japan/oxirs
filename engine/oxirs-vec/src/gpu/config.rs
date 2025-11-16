//! GPU configuration structures and enums

use serde::{Deserialize, Serialize};

/// Configuration for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub device_id: i32,
    pub enable_mixed_precision: bool,
    pub enable_tensor_cores: bool,
    pub batch_size: usize,
    pub memory_pool_size: usize,
    pub stream_count: usize,
    pub enable_peer_access: bool,
    pub enable_unified_memory: bool,
    pub enable_async_execution: bool,
    pub enable_multi_gpu: bool,
    pub preferred_gpu_ids: Vec<i32>,
    pub dynamic_batch_sizing: bool,
    pub enable_memory_compression: bool,
    pub kernel_cache_size: usize,
    pub optimization_level: OptimizationLevel,
    pub precision_mode: PrecisionMode,
}

/// GPU optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,       // Maximum debugging, minimal optimization
    Balanced,    // Good balance of performance and debugging
    Performance, // Maximum performance, minimal debugging
    Extreme,     // Aggressive optimizations, may reduce precision
}

/// Precision modes for GPU computations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionMode {
    FP32,     // Single precision
    FP16,     // Half precision
    Mixed,    // Mixed precision (FP16 for compute, FP32 for storage)
    INT8,     // 8-bit integer quantization
    Adaptive, // Adaptive precision based on data characteristics
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            batch_size: 1024,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            stream_count: 4,
            enable_peer_access: false,
            enable_unified_memory: false,
            enable_async_execution: true,
            enable_multi_gpu: false,
            preferred_gpu_ids: vec![0],
            dynamic_batch_sizing: true,
            enable_memory_compression: false,
            kernel_cache_size: 100, // Cache up to 100 compiled kernels
            optimization_level: OptimizationLevel::Balanced,
            precision_mode: PrecisionMode::FP32,
        }
    }
}

impl GpuConfig {
    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            optimization_level: OptimizationLevel::Performance,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            enable_async_execution: true,
            batch_size: 2048,
            stream_count: 8,
            ..Default::default()
        }
    }

    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            enable_memory_compression: true,
            enable_unified_memory: true,
            batch_size: 512,
            memory_pool_size: 512 * 1024 * 1024, // 512MB
            ..Default::default()
        }
    }

    /// Create a debug-friendly configuration
    pub fn debug() -> Self {
        Self {
            optimization_level: OptimizationLevel::Debug,
            enable_mixed_precision: false,
            enable_async_execution: false,
            batch_size: 64,
            stream_count: 1,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be greater than 0"));
        }
        if self.stream_count == 0 {
            return Err(anyhow::anyhow!("Stream count must be greater than 0"));
        }
        if self.memory_pool_size == 0 {
            return Err(anyhow::anyhow!("Memory pool size must be greater than 0"));
        }
        if self.kernel_cache_size == 0 {
            return Err(anyhow::anyhow!("Kernel cache size must be greater than 0"));
        }
        if self.preferred_gpu_ids.is_empty() {
            return Err(anyhow::anyhow!(
                "Must specify at least one preferred GPU ID"
            ));
        }
        Ok(())
    }

    /// Calculate optimal batch size based on available memory
    pub fn calculate_optimal_batch_size(
        &self,
        vector_dim: usize,
        available_memory: usize,
    ) -> usize {
        let bytes_per_vector = vector_dim * std::mem::size_of::<f32>();
        let max_vectors = available_memory / bytes_per_vector / 4; // Reserve 75% for safety
        max_vectors
            .min(self.batch_size * 4)
            .max(self.batch_size / 4)
    }
}
