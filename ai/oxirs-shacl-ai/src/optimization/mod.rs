//! Validation optimization and performance tuning
//!
//! This module implements AI-powered optimization for SHACL validation performance,
//! shape optimization, and validation strategy improvements.

pub mod config;
pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types and structs
pub use config::*;
pub use engine::*;
pub use types::{
    BottleneckSeverity, BottleneckType, CachePartitioningStrategy, CacheReplacementPolicy,
    CacheStrategy, ComplexityLevel, ConnectivityAnalysis, GcStrategy, GraphAnalysis,
    GraphStatistics, ImplementationEffort, LoadBalancingStrategy, MemoryOptimization, MemoryPool,
    OpportunityType, ValidationOptimizationOpportunity, OptimizationRecommendation,
    OptimizationRecommendationType, OptimizationStatistics, OptimizationTrainingData,
    OptimizedValidationStrategy, ParallelExecutionStrategy, PerformanceBottleneck,
    PerformanceImprovements, PoolType, RecommendationPriority, ShapeExecutionPlan,
    SynchronizationPoint,
};
