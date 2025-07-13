//! Configuration and core types for validation performance optimization
//!
//! This module contains configuration structures and basic types used throughout
//! the validation performance optimization system.

use serde::{Deserialize, Serialize};

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel validation
    pub enable_parallel_validation: bool,
    /// Number of worker threads for validation
    pub worker_threads: usize,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Enable constraint ordering optimization
    pub enable_constraint_ordering: bool,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit (number of entries)
    pub cache_size_limit: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable query optimization
    pub enable_query_optimization: bool,
    /// Enable index hints
    pub enable_index_hints: bool,
    /// Memory pool size in MB
    pub memory_pool_size_mb: usize,
    /// Resource allocation strategy
    pub resource_allocation: ResourceAllocationStrategy,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel_validation: true,
            worker_threads: num_cpus::get(),
            batch_size: 1000,
            enable_constraint_ordering: true,
            enable_caching: true,
            cache_size_limit: 10000,
            cache_ttl_seconds: 3600,
            enable_query_optimization: true,
            enable_index_hints: true,
            memory_pool_size_mb: 512,
            resource_allocation: ResourceAllocationStrategy::Adaptive,
        }
    }
}

/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAllocationStrategy {
    /// Static allocation based on configuration
    Static,
    /// Dynamic allocation based on workload
    Dynamic,
    /// Adaptive allocation with learning
    Adaptive,
    /// Round-robin allocation
    RoundRobin,
    /// Load-balanced allocation
    LoadBalanced,
}

/// Optimization strategies for constraint ordering
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Order by selectivity (most selective first)
    Selectivity,
    /// Order by execution cost (cheapest first)
    Cost,
    /// Order by dependency requirements
    Dependency,
    /// Hybrid approach combining multiple factors
    Hybrid,
    /// Machine learning-based ordering
    MachineLearning,
    /// Genetic algorithm optimization
    Genetic,
}

/// Task priority levels for parallel execution
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Performance improvement calculation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    pub optimization_type: String,
    pub baseline_time_ms: f64,
    pub optimized_time_ms: f64,
    pub improvement_percentage: f64,
    pub memory_reduction_mb: f64,
    pub cache_hit_rate_improvement: f64,
}

/// Performance-related statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub failed_validations: usize,
    pub average_validation_time_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization_percent: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            failed_validations: 0,
            average_validation_time_ms: 0.0,
            cache_hit_rate: 0.0,
            memory_usage_mb: 0.0,
            cpu_utilization_percent: 0.0,
        }
    }
}
