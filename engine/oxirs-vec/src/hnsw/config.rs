//! Configuration types for HNSW index

use crate::similarity::SimilarityMetric;
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use crate::gpu::GpuConfig;

/// Configuration for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of bi-directional links created for each node during construction (except layer 0)
    pub m: usize,
    /// Maximum number of bi-directional links created for each node during construction for layer 0
    pub m_l0: usize,
    /// Level generation factor
    pub ml: f64,
    /// Size of the dynamic candidate list
    pub ef: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Similarity metric to use
    pub metric: SimilarityMetric,
    /// Enable SIMD optimizations for distance calculations
    pub enable_simd: bool,
    /// Enable memory prefetching for improved cache performance
    pub enable_prefetch: bool,
    /// Enable parallel search across multiple threads
    pub enable_parallel: bool,
    /// Prefetch distance (number of nodes to prefetch ahead)
    pub prefetch_distance: usize,
    /// Enable cache-friendly data layout
    pub cache_friendly_layout: bool,
    /// Enable GPU acceleration for distance calculations and search
    pub enable_gpu: bool,
    /// GPU configuration (only used if enable_gpu is true)
    #[cfg(feature = "gpu")]
    pub gpu_config: Option<GpuConfig>,
    /// Minimum batch size for GPU operations (smaller batches use CPU)
    pub gpu_batch_threshold: usize,
    /// Enable multi-GPU support for massive datasets
    pub enable_multi_gpu: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_l0: 32,
            ml: 1.0 / (2.0_f64).ln(),
            ef: 50,
            ef_construction: 200,
            metric: SimilarityMetric::Cosine,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 8,
            cache_friendly_layout: true,
            enable_gpu: false,
            #[cfg(feature = "gpu")]
            gpu_config: None,
            gpu_batch_threshold: 1000,
            enable_multi_gpu: false,
        }
    }
}

impl HnswConfig {
    /// Create a performance-optimized configuration
    pub fn optimized() -> Self {
        Self {
            m: 32,
            m_l0: 64,
            ef: 100,
            ef_construction: 400,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 16,
            cache_friendly_layout: true,
            enable_gpu: false,
            #[cfg(feature = "gpu")]
            gpu_config: None,
            gpu_batch_threshold: 500,
            enable_multi_gpu: false,
            ..Default::default()
        }
    }

    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            m: 8,
            m_l0: 16,
            ef: 32,
            ef_construction: 100,
            enable_simd: true,
            enable_prefetch: false,
            enable_parallel: false,
            prefetch_distance: 4,
            cache_friendly_layout: true,
            enable_gpu: false,
            #[cfg(feature = "gpu")]
            gpu_config: None,
            gpu_batch_threshold: 2000,
            enable_multi_gpu: false,
            ..Default::default()
        }
    }

    /// Create a GPU-accelerated configuration for maximum performance
    #[cfg(feature = "gpu")]
    pub fn gpu_optimized() -> Self {
        Self {
            m: 48,
            m_l0: 96,
            ef: 200,
            ef_construction: 800,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 32,
            cache_friendly_layout: true,
            enable_gpu: true,
            gpu_config: Some(GpuConfig {
                enable_mixed_precision: true,
                enable_tensor_cores: true,
                batch_size: 4096,
                stream_count: 8,
                enable_async_execution: true,
                optimization_level: crate::gpu::OptimizationLevel::Performance,
                ..Default::default()
            }),
            gpu_batch_threshold: 100,
            enable_multi_gpu: false,
            ..Default::default()
        }
    }

    /// Create a multi-GPU configuration for massive datasets
    #[cfg(feature = "gpu")]
    pub fn multi_gpu_optimized(gpu_ids: Vec<i32>) -> Self {
        Self {
            m: 64,
            m_l0: 128,
            ef: 300,
            ef_construction: 1200,
            enable_simd: true,
            enable_prefetch: true,
            enable_parallel: true,
            prefetch_distance: 64,
            cache_friendly_layout: true,
            enable_gpu: true,
            gpu_config: Some(GpuConfig {
                enable_mixed_precision: true,
                enable_tensor_cores: true,
                batch_size: 8192,
                stream_count: 16,
                enable_async_execution: true,
                enable_multi_gpu: true,
                preferred_gpu_ids: gpu_ids,
                optimization_level: crate::gpu::OptimizationLevel::Performance,
                ..Default::default()
            }),
            gpu_batch_threshold: 50,
            enable_multi_gpu: true,
            ..Default::default()
        }
    }
}
