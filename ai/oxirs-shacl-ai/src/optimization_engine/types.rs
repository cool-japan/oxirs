//! Supporting types for the optimization engine

use crate::{
    shape::AiShape,
    shape_management::PerformanceProfile,
};
use std::collections::HashMap;
use std::time::Duration;

use super::config::OptimizationConfig;

/// Optimized shape result
#[derive(Debug, Clone)]
pub struct OptimizedShape {
    pub original_shape: AiShape,
    pub optimized_shape: AiShape,
    pub performance_profile: PerformanceProfile,
    pub applied_optimizations: Vec<OptimizationResult>,
    pub before_metrics: PerformanceMetrics,
    pub after_metrics: PerformanceMetrics,
    pub improvement_percentage: f64,
    pub optimization_metadata: OptimizationMetadata,
}

/// Optimization metadata
#[derive(Debug, Clone)]
pub struct OptimizationMetadata {
    pub optimized_at: chrono::DateTime<chrono::Utc>,
    pub optimization_duration: Duration,
    pub engine_version: String,
    pub configuration: OptimizationConfig,
}

/// Parallel validation configuration
#[derive(Debug, Clone)]
pub struct ParallelValidationConfig {
    pub enabled: bool,
    pub max_parallel_constraints: usize,
    pub constraint_groups: Vec<ConstraintGroup>,
    pub execution_strategy: ParallelExecutionStrategy,
    pub estimated_speedup: f64,
}

/// Parallelization plan
#[derive(Debug, Clone)]
pub struct ParallelizationPlan {
    pub constraint_groups: Vec<ConstraintGroup>,
    pub execution_strategy: ParallelExecutionStrategy,
    pub estimated_speedup: f64,
}

/// Constraint group for parallel execution
#[derive(Debug, Clone)]
pub struct ConstraintGroup {
    pub group_id: String,
    pub property_path: String,
    pub constraints: Vec<ConstraintReference>,
    pub parallel_safe: bool,
}

/// Reference to a constraint within a group
#[derive(Debug, Clone)]
pub struct ConstraintReference {
    pub index: usize,
    pub constraint_type: String,
    pub estimated_cost: f64,
}

/// Parallel execution strategy
#[derive(Debug, Clone)]
pub enum ParallelExecutionStrategy {
    Sequential,
    GroupBased,
    FullParallel,
    Adaptive,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfiguration {
    pub enabled: bool,
    pub cacheable_constraints: Vec<CacheableConstraint>,
    pub cache_strategies: Vec<CacheStrategy>,
    pub estimated_hit_rate: f64,
    pub memory_limit_mb: f64,
}

/// Cacheable constraint
#[derive(Debug, Clone)]
pub struct CacheableConstraint {
    pub constraint_index: usize,
    pub constraint_type: String,
    pub cache_key_strategy: CacheKeyStrategy,
    pub estimated_cache_hit_rate: f64,
}

/// Cache key strategy
#[derive(Debug, Clone)]
pub enum CacheKeyStrategy {
    PropertyBased,
    ValueBased,
    QueryBased,
    TypeBased,
    Composite,
}

/// Cache strategy
#[derive(Debug, Clone)]
pub struct CacheStrategy {
    pub strategy_name: String,
    pub strategy_type: CacheStrategyType,
    pub ttl_seconds: u64,
    pub eviction_policy: CacheEvictionPolicy,
}

/// Cache strategy type
#[derive(Debug, Clone)]
pub enum CacheStrategyType {
    ResultCaching,
    QueryCaching,
    DataCaching,
    MetadataCaching,
}

/// Cache eviction policy
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
    Adaptive,
}

/// Performance metrics for optimization analysis
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub validation_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub cache_hit_rate: f64,
    pub parallelization_factor: f64,
    pub constraint_execution_times: HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            validation_time_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            cache_hit_rate: 0.0,
            parallelization_factor: 1.0,
            constraint_execution_times: HashMap::new(),
        }
    }
}

/// Result of an optimization operation
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimization_type: String,
    pub before_performance: PerformanceMetrics,
    pub after_performance: PerformanceMetrics,
    pub improvement_percentage: f64,
    pub applied_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Statistics for optimization operations
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_optimizations: usize,
    pub successful_optimizations: usize,
    pub average_improvement: f64,
    pub optimization_history: Vec<OptimizationResult>,
    pub last_optimization: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for OptimizationStatistics {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            successful_optimizations: 0,
            average_improvement: 0.0,
            optimization_history: Vec::new(),
            last_optimization: None,
        }
    }
}
