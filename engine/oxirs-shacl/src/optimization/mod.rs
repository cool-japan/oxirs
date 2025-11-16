//! SHACL Validation Performance Optimization
//!
//! This module provides comprehensive performance optimization capabilities for SHACL validation,
//! including constraint evaluation caching, parallel processing, incremental validation,
//! and streaming validation for large datasets.

pub mod advanced_batch;
pub mod advanced_performance;
pub mod constraint_ordering;
pub mod core;
pub mod gpu_accelerated;
pub mod gpu_validation;
pub mod integration;
pub mod memory;
pub mod negation_optimizer;
pub mod parallel;
pub mod quantum_analytics;
pub mod scirs2_memory;
pub mod scirs2_parallel;
pub mod scirs2_simd;

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
    compact::{CompactConverter, CompactShape, CompactViolation},
    InternedString, MemoryMonitor, MemoryOptimizationConfig, MemoryOptimizationStats,
    MemoryOptimizer, MemoryPressureLevel, MemoryTrend, OptimizationResult, StringInterner,
};

// Re-export quantum analytics types
pub use quantum_analytics::{
    ConsciousnessInsight, ConsciousnessState, EntanglementInfo, EntanglementType, MeditationState,
    QuantumAnalyticsConfig, QuantumOptimizationStrategy, QuantumPerformanceAnalytics,
    QuantumPerformanceInsight, RecommendationCategory, TranscendenceComplexity,
    TranscendentRecommendation,
};

// Re-export negation optimizer types
pub use negation_optimizer::{
    CacheWorthiness, NegationOptimizationConfig, NegationOptimizationResult,
    NegationOptimizationStats, NegationOptimizer, OptimizationStrategy, ShapeComplexity,
    ShapeComplexityAnalysis, StrategyStats,
};

// Re-export SciRS2 parallel validation types
pub use scirs2_parallel::{
    ParallelValidationResult as SciRS2ParallelValidationResult, SciRS2ParallelConfig,
    SciRS2ParallelValidator,
};

// Re-export SciRS2 memory-efficient validation types
pub use scirs2_memory::{
    AdaptiveChunking, LazyEvaluationCache, MemoryEfficientValidationResult, MemoryMetrics,
    SciRS2MemoryConfig, SciRS2MemoryValidator,
};

// Re-export SciRS2 SIMD-accelerated validation types
pub use scirs2_simd::{
    SimdAccelerationConfig, SimdConstraintValidator, SimdPerformanceMetrics, SimdValidationResult,
};
