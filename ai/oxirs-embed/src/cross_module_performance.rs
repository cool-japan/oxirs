//! Cross-Module Performance — facade
//!
//! Re-exports all public items from the sibling modules.

pub use crate::cross_module_performance_profiler::{
    calculate_anomaly_score, calculate_percentage_change, AnomalyDetector, LearningEngine,
    ModulePerformanceMonitor, PredictivePerformanceEngine, ResourceAllocator, ResourceTracker,
};
pub use crate::cross_module_performance_reporter::{
    CrossModulePerformanceCoordinator, GlobalPerformanceMetrics, OptimizationCache,
};
pub use crate::cross_module_performance_types::{
    AllocationEvent, AllocationStrategy, AllocationType, AnomalyAlgorithm, AnomalyEvent,
    AnomalyType, CacheStats, CachedOptimization, CachedPrediction, CoordinatorConfig, ModelType,
    ModuleMetrics, OptimizationRecommendation, OptimizationResults, OptimizationType,
    PerformanceBaseline, PerformanceImpact, PerformanceModel, PerformanceSnapshot, Priority,
    ResourceAllocation, SeverityLevel, TrainingSample,
};
