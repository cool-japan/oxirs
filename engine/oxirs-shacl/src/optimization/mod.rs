//! SHACL Validation Performance Optimization
//!
//! This module provides comprehensive performance optimization capabilities for SHACL validation,
//! including constraint evaluation caching, parallel processing, incremental validation,
//! and streaming validation for large datasets.

pub mod advanced_batch;
pub mod advanced_performance;
pub mod constraint_ordering;
pub mod core;
pub mod integration;
pub mod memory;
pub mod parallel;

// Re-export key types from core
pub use core::{
    AdvancedConstraintEvaluator, BatchConstraintEvaluator, CacheStats, ConstraintCache,
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
    ParallelValidationConfig, ParallelValidationEngine, ParallelValidationResult, WorkerPoolStats,
};

// Re-export advanced batch types
pub use advanced_batch::{
    AdvancedBatchConfig, AdvancedBatchValidator, BatchExecutionStats, BatchValidationResult,
    MemoryUsageMonitor,
};

// Re-export constraint ordering types
pub use constraint_ordering::{
    ConstraintOrderingResult, ConstraintOrderingStats, ConstraintSelectivityAnalyzer,
    EarlyTerminationStrategy, OrderedConstraint, SelectivityConfig,
};

// Re-export advanced performance analytics types
pub use advanced_performance::{
    AdvancedPerformanceAnalytics, AnalyticsConfig, ConstraintPerformanceMetrics,
    GlobalPerformanceMetrics, PerformanceAlert, PerformanceHealth, PerformancePrediction,
    PerformanceReport, PerformanceStatus, PerformanceTrend, ShapePerformanceMetrics,
};

// Re-export memory optimization types
pub use memory::{
    MemoryOptimizationConfig, MemoryOptimizationStats, MemoryOptimizer, MemoryMonitor,
    MemoryPressureLevel, MemoryTrend, StringInterner, InternedString, OptimizationResult,
    compact::{CompactViolation, CompactShape, CompactConverter},
};
