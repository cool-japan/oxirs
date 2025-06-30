//! SHACL Validation Performance Optimization
//!
//! This module provides comprehensive performance optimization capabilities for SHACL validation,
//! including constraint evaluation caching, parallel processing, incremental validation,
//! and streaming validation for large datasets.

pub mod core;
pub mod integration;
pub mod parallel;

// Re-export key types from core
pub use core::{
    AdvancedConstraintEvaluator, BatchConstraintEvaluator, ConstraintCache, CacheStats,
    ConstraintDependencyAnalyzer, ConstraintPerformanceStats, IncrementalValidationEngine,
    IncrementalValidationResult, IncrementalValidationStats, OptimizationConfig,
    OptimizationMetrics, StreamingValidationEngine, StreamingValidationResult,
    ValidationOptimizationEngine,
};

// Re-export integration types
pub use integration::{
    CacheStatistics, OptimizationFeature, OptimizedValidationEngine, ValidationContext,
    ValidationPerformanceMetrics, ValidationStrategy,
};

// Re-export parallel types
pub use parallel::{
    ParallelValidationConfig, ParallelValidationEngine, ParallelValidationResult,
    WorkerPoolStats,
};