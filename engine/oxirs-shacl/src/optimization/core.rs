//! SHACL Constraint Evaluation Optimizations — Facade
//!
//! Re-exports all optimization types. Implementation details live in:
//!
//! - `super::core_engine` — ConstraintCache, BatchConstraintEvaluator,
//!   ValidationOptimizationEngine + config/metrics
//! - `super::core_strategies` — ConstraintDependencyAnalyzer, AdvancedConstraintEvaluator,
//!   StreamingValidationEngine, IncrementalValidationEngine,
//!   and all change/delta types

#![allow(dead_code)]

// Engine types
pub use super::core_engine::{
    BatchConstraintEvaluator, CacheStats, ConstraintCache, OptimizationConfig, OptimizationMetrics,
    ValidationOptimizationEngine,
};

// Strategy and analysis types
pub use super::core_strategies::{
    AdvancedConstraintEvaluator, ChangeDetectionLevel, ChangeEvent, ChangeEventType, ChangesDelta,
    ConstraintDependencyAnalyzer, ConstraintPerformanceStats, IncrementalValidationEngine,
    IncrementalValidationResult, IncrementalValidationStats, NodeConstraintChange,
    NodePropertyChange, PropertyChange, PropertyChangeType, StreamingValidationEngine,
    StreamingValidationResult,
};
