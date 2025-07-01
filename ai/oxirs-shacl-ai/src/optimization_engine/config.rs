//! Configuration for the optimization engine

use serde::{Deserialize, Serialize};

/// Configuration for optimization engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable parallel validation
    pub enable_parallel_validation: bool,

    /// Maximum number of parallel validation threads
    pub max_parallel_threads: usize,

    /// Enable constraint result caching
    pub enable_constraint_caching: bool,

    /// Cache size limit (number of entries)
    pub cache_size_limit: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Enable constraint ordering optimization
    pub enable_constraint_ordering: bool,

    /// Enable dynamic optimization
    pub enable_dynamic_optimization: bool,

    /// Performance threshold for optimization triggers (ms)
    pub performance_threshold_ms: f64,

    /// Memory usage threshold for optimization (MB)
    pub memory_threshold_mb: f64,

    /// Enable sophisticated profiling
    pub enable_profiling: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_parallel_validation: true,
            max_parallel_threads: num_cpus::get().min(8),
            enable_constraint_caching: true,
            cache_size_limit: 10000,
            cache_ttl_seconds: 3600, // 1 hour
            enable_constraint_ordering: true,
            enable_dynamic_optimization: true,
            performance_threshold_ms: 100.0,
            memory_threshold_mb: 512.0,
            enable_profiling: true,
        }
    }
}
