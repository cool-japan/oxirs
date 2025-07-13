//! Validation Performance Optimization Module
//!
//! This module provides comprehensive performance optimization capabilities for SHACL validation,
//! including constraint ordering, parallel execution, caching, resource monitoring, and query optimization.

pub mod cache;
pub mod config;
pub mod constraint_optimizer;
pub mod index_optimizer;
pub mod parallel_executor;
pub mod query_optimizer;
pub mod resource_monitor;
pub mod types;

// Re-export key types and structs for easier access
pub use cache::{CacheEfficiencyMetrics, ValidationCache};
pub use config::{
    OptimizationStrategy, PerformanceConfig, ResourceAllocationStrategy, TaskPriority,
    ViolationSeverity,
};
pub use constraint_optimizer::ConstraintOrderOptimizer;
pub use index_optimizer::{
    IndexCreationCost, IndexOptimizer, IndexRecommendation, IndexType, IndexUsageStats,
};
pub use parallel_executor::{ParallelPerformanceStats, ParallelValidationExecutor};
pub use query_optimizer::{
    OptimizationTechnique, QueryComplexity, QueryOptimization, QueryOptimizer,
};
pub use resource_monitor::{
    AlertSeverity, AlertType, ResourceAlert, ResourceMonitor, ResourceTrends,
};
pub use types::{
    CacheStatistics, CachedValidationResult, ConsciousnessState, ConstraintDependencyGraph,
    ConstraintPerformanceStats, IndexUsagePatterns, NeuralPatternResult, OptimizationContext,
    OptimizationRecommendation, ParallelExecutionStats, PerformanceRequirements,
    QuantumPerformanceMetrics, QueryPerformanceMetrics, ResourceThresholds, ResourceUsage,
    ValidationResult, ValidationTask, ValidationViolation,
};

use crate::{ShaclAiError, Shape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main performance optimization manager
#[derive(Debug, Clone)]
pub struct ValidationPerformanceOptimizer {
    config: PerformanceConfig,
    constraint_optimizer: ConstraintOrderOptimizer,
    parallel_executor: ParallelValidationExecutor,
    index_optimizer: IndexOptimizer,
    query_optimizer: QueryOptimizer,
}

/// Optimized validation plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedValidationPlan {
    pub optimized_constraints: HashMap<String, Vec<String>>,
    pub parallel_plan: ParallelExecutionPlan,
    pub query_optimizations: Vec<QueryOptimization>,
    pub index_recommendations: Vec<IndexRecommendation>,
    pub resource_plan: ResourceAllocationPlan,
    pub estimated_performance_improvement: PerformanceImprovement,
}

/// Parallel execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionPlan {
    pub worker_threads: usize,
    pub batch_size: usize,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub resource_allocation: ResourceAllocationStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    Priority,
    Adaptive,
}

/// Resource allocation plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationPlan {
    pub memory_allocation_mb: usize,
    pub cpu_allocation_percent: f64,
    pub io_bandwidth_mbps: f64,
    pub cache_allocation_mb: usize,
    pub temporary_storage_mb: usize,
}

/// Performance improvement estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    pub baseline_execution_time_ms: f64,
    pub optimized_execution_time_ms: f64,
    pub improvement_percentage: f64,
    pub throughput_improvement: f64,
    pub memory_reduction_percentage: f64,
    pub cpu_efficiency_improvement: f64,
}

impl ValidationPerformanceOptimizer {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            constraint_optimizer: ConstraintOrderOptimizer::new(OptimizationStrategy::Hybrid),
            parallel_executor: ParallelValidationExecutor::new(config.clone()),
            index_optimizer: IndexOptimizer::new(),
            query_optimizer: QueryOptimizer::new(),
            config,
        }
    }

    /// Create a new optimizer with custom optimization strategy
    pub fn with_strategy(config: PerformanceConfig, strategy: OptimizationStrategy) -> Self {
        Self {
            constraint_optimizer: ConstraintOrderOptimizer::new(strategy),
            parallel_executor: ParallelValidationExecutor::new(config.clone()),
            index_optimizer: IndexOptimizer::new(),
            query_optimizer: QueryOptimizer::new(),
            config,
        }
    }

    /// Optimize validation performance for a set of shapes
    pub fn optimize_validation_performance(
        &self,
        shapes: &[Shape],
    ) -> Result<OptimizedValidationPlan, ShaclAiError> {
        // Step 1: Optimize constraint ordering
        let optimized_constraints = self.optimize_constraint_execution_order(shapes)?;

        // Step 2: Create parallel execution plan
        let parallel_plan = self.create_parallel_execution_plan(shapes)?;

        // Step 3: Optimize queries and indexes
        let query_optimizations = self.query_optimizer.optimize_queries(shapes)?;
        let index_recommendations = self.index_optimizer.recommend_indexes(shapes)?;

        // Step 4: Resource allocation plan
        let resource_plan = self.create_resource_allocation_plan(shapes)?;

        Ok(OptimizedValidationPlan {
            optimized_constraints,
            parallel_plan,
            query_optimizations,
            index_recommendations,
            resource_plan,
            estimated_performance_improvement: self.estimate_performance_improvement(shapes)?,
        })
    }

    /// Optimize constraint execution order
    fn optimize_constraint_execution_order(
        &self,
        shapes: &[Shape],
    ) -> Result<HashMap<String, Vec<String>>, ShaclAiError> {
        let mut optimized_constraints = HashMap::new();

        for shape in shapes {
            let constraint_ids: Vec<String> = shape
                .property_constraints
                .iter()
                .map(|c| c.path.clone())
                .collect();
            let optimized_order = self
                .constraint_optimizer
                .optimize_constraint_order(&constraint_ids)?;
            optimized_constraints.insert(shape.id.to_string(), optimized_order);
        }

        Ok(optimized_constraints)
    }

    /// Create parallel execution plan
    fn create_parallel_execution_plan(
        &self,
        shapes: &[Shape],
    ) -> Result<ParallelExecutionPlan, ShaclAiError> {
        let total_constraints: usize = shapes.iter().map(|s| s.property_constraints.len()).sum();
        let optimal_parallelism = self.calculate_optimal_parallelism(total_constraints);

        Ok(ParallelExecutionPlan {
            worker_threads: optimal_parallelism,
            batch_size: self.config.batch_size,
            load_balancing_strategy: LoadBalancingStrategy::WorkStealing,
            resource_allocation: self.config.resource_allocation.clone(),
        })
    }

    /// Calculate optimal parallelism level
    fn calculate_optimal_parallelism(&self, total_work: usize) -> usize {
        let available_cores = num_cpus::get();
        let work_per_core = total_work / available_cores.max(1);

        if work_per_core < 10 {
            // Low work, use fewer cores to reduce overhead
            (available_cores / 2).max(1)
        } else if work_per_core > 1000 {
            // High work, use all available cores
            available_cores
        } else {
            // Medium work, use 75% of cores
            (available_cores * 3 / 4).max(1)
        }
    }

    /// Create resource allocation plan
    fn create_resource_allocation_plan(
        &self,
        shapes: &[Shape],
    ) -> Result<ResourceAllocationPlan, ShaclAiError> {
        let estimated_memory_mb = shapes.len() * 50; // 50MB per shape estimate
        let estimated_cpu_percent = 80.0; // Target 80% CPU utilization

        Ok(ResourceAllocationPlan {
            memory_allocation_mb: estimated_memory_mb,
            cpu_allocation_percent: estimated_cpu_percent,
            io_bandwidth_mbps: 100.0,
            cache_allocation_mb: self.config.memory_pool_size_mb / 4,
            temporary_storage_mb: self.config.memory_pool_size_mb / 2,
        })
    }

    /// Estimate performance improvement
    fn estimate_performance_improvement(
        &self,
        shapes: &[Shape],
    ) -> Result<PerformanceImprovement, ShaclAiError> {
        let baseline_time_ms = shapes.len() as f64 * 100.0; // 100ms per shape baseline

        // Calculate improvements from various optimizations
        let constraint_ordering_improvement = 0.25; // 25% improvement
        let parallel_execution_improvement = 0.40; // 40% improvement
        let caching_improvement = 0.15; // 15% improvement
        let query_optimization_improvement = 0.20; // 20% improvement

        let total_improvement = 1.0
            - (1.0 - constraint_ordering_improvement)
                * (1.0 - parallel_execution_improvement)
                * (1.0 - caching_improvement)
                * (1.0 - query_optimization_improvement);

        let optimized_time_ms = baseline_time_ms * (1.0 - total_improvement);

        Ok(PerformanceImprovement {
            baseline_execution_time_ms: baseline_time_ms,
            optimized_execution_time_ms: optimized_time_ms,
            improvement_percentage: total_improvement * 100.0,
            throughput_improvement: 1.0 / (1.0 - total_improvement),
            memory_reduction_percentage: 20.0, // 20% memory reduction
            cpu_efficiency_improvement: 30.0,  // 30% CPU efficiency improvement
        })
    }

    /// Get current performance statistics
    pub fn get_performance_statistics(&self) -> ParallelPerformanceStats {
        self.parallel_executor.get_performance_stats()
    }

    /// Execute parallel validation with optimization
    pub fn execute_optimized_validation(
        &self,
        shapes: &[Shape],
    ) -> Result<Vec<ValidationResult>, ShaclAiError> {
        // Start resource monitoring
        self.parallel_executor.start_monitoring();

        // Execute with current configuration
        self.parallel_executor
            .execute_parallel_validation(shapes, None)
    }

    /// Update performance configuration
    pub fn update_config(&mut self, new_config: PerformanceConfig) {
        self.config = new_config.clone();
        self.parallel_executor = ParallelValidationExecutor::new(new_config);
    }

    /// Get optimizer configuration
    pub fn get_config(&self) -> &PerformanceConfig {
        &self.config
    }

    /// Get constraint optimizer
    pub fn get_constraint_optimizer(&self) -> &ConstraintOrderOptimizer {
        &self.constraint_optimizer
    }

    /// Get parallel executor
    pub fn get_parallel_executor(&self) -> &ParallelValidationExecutor {
        &self.parallel_executor
    }

    /// Get index optimizer
    pub fn get_index_optimizer(&self) -> &IndexOptimizer {
        &self.index_optimizer
    }

    /// Get query optimizer
    pub fn get_query_optimizer(&self) -> &QueryOptimizer {
        &self.query_optimizer
    }
}
