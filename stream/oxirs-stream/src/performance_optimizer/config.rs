//! Performance optimization configuration
//!
//! This module provides configuration structures for various performance optimization
//! features including adaptive batching, memory pooling, zero-copy operations, and parallel processing.

use serde::{Deserialize, Serialize};

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable adaptive batching
    pub enable_adaptive_batching: bool,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Target latency for adaptive batching
    pub target_latency_ms: u64,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Number of parallel workers
    pub parallel_workers: usize,
    /// Enable event pre-filtering
    pub enable_event_filtering: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
    /// Enable adaptive compression based on network conditions
    pub enable_adaptive_compression: bool,
    /// Network bandwidth estimation (bytes/sec)
    pub estimated_bandwidth: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_batching: true,
            max_batch_size: 1000,
            target_latency_ms: 10,
            enable_memory_pooling: true,
            memory_pool_size: 1024 * 1024 * 10, // 10MB
            enable_zero_copy: true,
            enable_parallel_processing: true,
            parallel_workers: num_cpus::get().max(1),
            enable_event_filtering: true,
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            enable_adaptive_compression: true,
            estimated_bandwidth: 100 * 1024 * 1024, // 100MB/s
        }
    }
}

/// Enhanced ML configuration for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMLConfig {
    /// Enable enhanced ML features
    pub enable_enhanced_ml: bool,
    /// Learning rate for neural network
    pub learning_rate: f64,
    /// Number of training epochs
    pub training_epochs: usize,
    /// Batch size for ML training
    pub ml_batch_size: usize,
    /// Neural network hidden layer size
    pub hidden_layer_size: usize,
    /// Enable feature engineering
    pub enable_feature_engineering: bool,
    /// Enable polynomial features
    pub enable_polynomial_features: bool,
    /// Polynomial degree for feature engineering
    pub polynomial_degree: usize,
    /// Enable interaction features
    pub enable_interaction_features: bool,
    /// Enable temporal features
    pub enable_temporal_features: bool,
    /// Enable statistical features
    pub enable_statistical_features: bool,
    /// Enable model selection
    pub enable_model_selection: bool,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Enable feature scaling
    pub enable_feature_scaling: bool,
    /// Model performance tracking window
    pub performance_window: usize,
}

impl Default for EnhancedMLConfig {
    fn default() -> Self {
        Self {
            enable_enhanced_ml: true,
            learning_rate: 0.01,
            training_epochs: 100,
            ml_batch_size: 32,
            hidden_layer_size: 64,
            enable_feature_engineering: true,
            enable_polynomial_features: true,
            polynomial_degree: 2,
            enable_interaction_features: true,
            enable_temporal_features: true,
            enable_statistical_features: true,
            enable_model_selection: true,
            cv_folds: 5,
            enable_feature_scaling: true,
            performance_window: 1000,
        }
    }
}

/// Configuration for batch size prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch size adjustment factor
    pub adjustment_factor: f64,
    /// Target latency for batch processing
    pub target_latency_ms: u64,
    /// Latency tolerance
    pub latency_tolerance_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 100,
            min_batch_size: 10,
            max_batch_size: 1000,
            adjustment_factor: 1.2,
            target_latency_ms: 10,
            latency_tolerance_ms: 2,
        }
    }
}

/// Configuration for memory pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Growth factor when expanding pool
    pub growth_factor: f64,
    /// Shrink threshold (percentage of unused memory)
    pub shrink_threshold: f64,
    /// Enable memory compaction
    pub enable_compaction: bool,
    /// Compaction interval (seconds)
    pub compaction_interval: u64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024,   // 1MB
            max_size: 100 * 1024 * 1024, // 100MB
            growth_factor: 1.5,
            shrink_threshold: 0.7,
            enable_compaction: true,
            compaction_interval: 60,
        }
    }
}

/// Configuration for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Queue capacity per worker
    pub queue_capacity: usize,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Enable thread pinning
    pub enable_thread_pinning: bool,
}

/// Load balancing strategies for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    Random,
    Weighted(Vec<f64>),
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get().max(1),
            queue_capacity: 1000,
            enable_work_stealing: true,
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            enable_thread_pinning: false,
        }
    }
}

/// Configuration for compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u32,
    /// Compression threshold (bytes)
    pub threshold: usize,
    /// Enable adaptive compression
    pub enable_adaptive: bool,
    /// Bandwidth threshold for compression (bytes/sec)
    pub bandwidth_threshold: u64,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
    Brotli,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            threshold: 1024, // 1KB
            enable_adaptive: true,
            bandwidth_threshold: 10 * 1024 * 1024, // 10MB/s
        }
    }
}
