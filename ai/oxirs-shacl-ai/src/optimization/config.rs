//! Configuration structures for optimization engine

use serde::{Deserialize, Serialize};

/// Configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable shape optimization
    pub enable_shape_optimization: bool,

    /// Enable validation strategy optimization
    pub enable_strategy_optimization: bool,

    /// Enable performance optimization
    pub enable_performance_optimization: bool,

    /// Enable parallel processing optimization
    pub enable_parallel_optimization: bool,

    /// Optimization algorithms
    pub algorithms: OptimizationAlgorithms,

    /// Performance targets
    pub performance_targets: PerformanceTargets,

    /// Enable training
    pub enable_training: bool,

    /// Optimization cache settings
    pub cache_settings: OptimizationCacheSettings,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_shape_optimization: true,
            enable_strategy_optimization: true,
            enable_performance_optimization: true,
            enable_parallel_optimization: true,
            algorithms: OptimizationAlgorithms::default(),
            performance_targets: PerformanceTargets::default(),
            enable_training: true,
            cache_settings: OptimizationCacheSettings::default(),
        }
    }
}

/// Optimization algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAlgorithms {
    /// Enable constraint reordering
    pub enable_constraint_reordering: bool,

    /// Enable shape merging
    pub enable_shape_merging: bool,

    /// Enable genetic algorithm
    pub enable_genetic_algorithm: bool,

    /// Enable simulated annealing
    pub enable_simulated_annealing: bool,

    /// Enable parallel optimization
    pub enable_parallel_optimization: bool,

    /// Enable machine learning optimization
    pub enable_ml_optimization: bool,
    
    /// Shape merge threshold (similarity threshold for merging shapes)
    pub shape_merge_threshold: f64,
}

impl Default for OptimizationAlgorithms {
    fn default() -> Self {
        Self {
            enable_constraint_reordering: true,
            enable_shape_merging: false,
            enable_genetic_algorithm: true,
            enable_simulated_annealing: false,
            enable_parallel_optimization: true,
            enable_ml_optimization: true,
            shape_merge_threshold: 0.8,
        }
    }
}

/// Performance targets for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target execution time (seconds)
    pub target_execution_time: f64,

    /// Target memory usage (MB)
    pub target_memory_mb: u64,

    /// Target CPU usage percentage
    pub target_cpu_percent: u8,

    /// Target throughput (validations per second)
    pub target_throughput: f64,

    /// Maximum acceptable latency (seconds)
    pub max_latency: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_execution_time: 5.0,
            target_memory_mb: 512,
            target_cpu_percent: 70,
            target_throughput: 100.0,
            max_latency: 10.0,
        }
    }
}

/// Optimization cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCacheSettings {
    /// Enable optimization result caching
    pub enable_caching: bool,

    /// Maximum cache size
    pub max_cache_size: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for OptimizationCacheSettings {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 500,
            cache_ttl_seconds: 1800, // 30 minutes
        }
    }
}
